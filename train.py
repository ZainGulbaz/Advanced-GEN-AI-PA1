import argparse
import math
from pathlib import Path
import torch

from src.model import TransformerLM
from src.optim import AdamW, get_lr_cosine_schedule, gradient_clipping, cross_entropy
from src.data import load_token_array, get_batch, save_checkpoint, load_checkpoint


def evaluate_validation(model, val_tokens, batch_size, sequence_length, device,
                        val_generator, num_validation_batches):
    if num_validation_batches <= 0:
        raise ValueError("num_validation_batches must be positive")
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.inference_mode():
            for _ in range(num_validation_batches):
                x, y = get_batch(val_tokens, batch_size, sequence_length, device, val_generator)
                total += cross_entropy(model(x), y).item()
    finally:
        model.train(was_training)
    return total / num_validation_batches


def parse_args():
    p = argparse.ArgumentParser(description="Train the PA1 Transformer LM on TinyStories.")
    p.add_argument("--train-data", default="data/tinystories/data/train.bin")
    p.add_argument("--val-data", default="data/tinystories/data/validation.bin")
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--n-q-heads", type=int, default=16)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--norm-eps", type=float, default=1e-5)
    p.add_argument("--lr-max", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cosine-steps", type=int, default=9999)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--num-validation-batches", type=int, default=20)
    p.add_argument("--train-seed", type=int, default=1337)
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def export_final_model(model, path="final_model.pt"):
    state = {
        name: tensor.detach().cpu().to(torch.float16) if tensor.is_floating_point()
        else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, path)


def main():
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if args.sequence_length != args.context_length:
        raise ValueError("assignment training uses sequence_length == context_length")
    if args.cosine_steps != args.num_steps - 1:
        raise ValueError("cosine_steps must equal num_steps - 1 for the final run")
    if args.eval_interval <= 0 or args.log_interval <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("evaluation, logging, and checkpoint intervals must be positive")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    train_tokens = load_token_array(args.train_data)
    val_tokens = load_token_array(args.val_data)
    train_gen = torch.Generator(device="cpu").manual_seed(args.train_seed)
    val_gen = torch.Generator(device="cpu").manual_seed(args.val_seed)

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers,
        args.n_q_heads, args.n_kv_heads, args.d_ff,
        args.rope_theta, args.norm_eps, device=device,
    )
    optimizer = AdamW(
        model.parameters(), lr=args.lr_max,
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    next_step = 0
    if args.resume:
        next_step = load_checkpoint(args.checkpoint, model, optimizer, train_gen, val_gen)
        print(f"Resumed from step {next_step}")

    print(f"Training on {device}; parameters={sum(p.numel() for p in model.parameters()):,}")
    for step in range(next_step, args.num_steps):
        model.train()
        lr = get_lr_cosine_schedule(
            step, args.lr_max, args.lr_min,
            args.warmup_steps, args.cosine_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(
                train_tokens, args.batch_size, args.sequence_length,
                device, train_gen,
            )
            microbatch_loss = cross_entropy(model(x), y)
            (microbatch_loss / args.gradient_accumulation_steps).backward()
            train_loss += microbatch_loss.detach().item()
        train_loss /= args.gradient_accumulation_steps

        grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        completed_steps = step + 1

        final_step = completed_steps == args.num_steps
        should_validate = final_step or completed_steps % args.eval_interval == 0
        should_log = should_validate or completed_steps % args.log_interval == 0
        should_checkpoint = final_step or completed_steps % args.checkpoint_interval == 0

        val_loss = None
        if should_validate:
            val_loss = evaluate_validation(
                model, val_tokens, args.batch_size, args.sequence_length,
                device, val_gen, args.num_validation_batches,
            )

        if should_log:
            msg = (
                f"step={completed_steps:5d} lr={lr:.3e} "
                f"train_loss={train_loss:.4f} grad_norm={grad_norm:.4f}"
            )
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}"
            print(msg)

        if should_checkpoint:
            save_checkpoint(
                model, optimizer, completed_steps,
                train_gen, val_gen, args.checkpoint,
            )

    export_final_model(model, "final_model.pt")
    print("Training complete. Exported final_model.pt")


if __name__ == "__main__":
    main()
