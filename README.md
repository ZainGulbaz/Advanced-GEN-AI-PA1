# Rectified PA1 Implementation

This is a corrected end-to-end implementation based on the supplied CS 5326 PA1 specification and the submitted code.

Run the assignment's public tests from the repository root:

```bash
uv run pytest
```

Download TinyStories using the assignment's Section 2.4 command, then train with:

```bash
uv run python train.py
```

Resume:

```bash
uv run python train.py --resume --checkpoint checkpoints/latest.pt
```

The final model artifact is exported as `final_model.pt` in the exact model-only state-dict format required by the assignment.
