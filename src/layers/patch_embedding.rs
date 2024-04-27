use burn::{
    tensor::backend::Backend,
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig}, prelude::*,
};

/*
For the positional embedding, we are given a tensor of shape (H, W, C) and we need to convert it to
(N, P^2*C), where P is the patch size, C is the number of channels, N = number of possible patches 
given by N = HW/P^2.

An way to solve this is to use a conv layer, map (B, C, H, W) to (B, HW/P^2, P^2*C).
    - The stride and kernel size will be the patch size.
    - We can use a convolution operation to  map it to the embedding dimension
    - The output will be (B, EMB, H/P, W/P), we can make this -> (B, EMB, HW/P^2)
    - We can permute it to get [B, HW/P^2, EMB].
*/

#[derive(Debug, Module)]
pub struct PatchEmbedding<B: Backend> {
    conv: Conv2d<B>,
}

#[derive(Debug, Config)]
pub struct PatchEmbeddingConfig {
    patch_size: usize,
    embedding_dim: usize, 
    in_channels: usize,
}

impl PatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedding<B> {
        PatchEmbedding {
            conv: Conv2dConfig::new(
                [self.in_channels, self.embedding_dim], 
                [self.patch_size, self.patch_size]
            ).with_stride([self.patch_size, self.patch_size]).init(device)
        }
    }
}

impl<B: Backend> PatchEmbedding<B> {
    // Input: [B, H, W, C]
    // Output: [B, X, Y[]
    pub fn forward(&self, inputs: Tensor<B, 4>) -> Tensor<B, 3> {
        let x: Tensor<B, 4> = self.conv.forward(inputs);
        // if we have a 224, 224 image, then the output will be 224/p, 224/p, for these particular parameters.
        let [batch_size, embedding_dim, height, width] = x.dims();
        let x: Tensor<B, 3> = x.reshape([batch_size, embedding_dim, height*width]);
        let x: Tensor<B, 3> = x.permute([0, 2, 1]);
        return x;
    }
}
