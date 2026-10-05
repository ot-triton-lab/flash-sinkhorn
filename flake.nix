{
  description = "Hugging Face Kernels build for FlashSinkhorn";

  inputs = {
    # Use the same builder revision as the build workflow.
    kernel-builder.url = "github:huggingface/kernels/c212e38db005a95ef905858bdeca7a3c15b607a4";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
