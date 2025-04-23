# ICL with Mamba and Transformer: Trained Model Links

This directory contains trained models comparing the **Transformer** and **Mamba** architectures on two key in-context learning tasks: **Sinusoidal Regression** and **Long-Term Dependency**.

The reason why we uploaded the models to HuggingFace is because we had some difficulties with uploading our `model_100000.pt` files due to Git LFS and forked repo problems.

## Model Links

| Task ↓ / Model → | Transformer | Mamba |
|------------------|-------------|--------|
| Sinusoidal Regression | [🤗 View on Hugging Face](https://huggingface.co/kkodnad/sinusoidal_regression_tf_embd512_layer8_lr1e-4/tree/main) | [🤗 View on Hugging Face](https://huggingface.co/kkodnad/sinusoidal_regression_mamba_embd512_layer8_lr1e-4/tree/main) |
| Long Term Dependency | [🤗 View on Hugging Face](https://huggingface.co/kkodnad/long_term_dependency_gpt2_embd512_layer8_lr1e-4/tree/main) | [🤗 View on Hugging Face](https://huggingface.co/kkodnad/long_term_dependency_mamba_embd512_layer8_lr1e-4/tree/main) |
