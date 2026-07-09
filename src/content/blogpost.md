---
title: Llamas on the Web
subtitle: Introducing WebGPU support for llama.cpp
byline: May 2026
---

:::links
📄 [Paper](https://arxiv.org/abs/2605.20706)
💻 [GitHub](https://github.com/ggml-org/llama.cpp)
:::

After nearly a year of work, we're officially introducing WebGPU support for llama.cpp. Now, users can now run almost any open-weight model in their browser, _with GPU acceleration_, making fast and local LLM inference more accessible and available than it ever has been before.

Wondering what browser-based inference can look like in practice? Go ahead and try it out below:

:::demo

## Why Use Local Models in the Browser?

You might be impressed or underwhelmed by the demo above, depending on your experience using the powerful LLMs served by big companies like OpenAI or Anthropic. However, the capabilities of smaller language models are [improving rapidly](https://hazyresearch.stanford.edu/blog/2025-11-11-ipw) and they may play a large part in the [future of agentic AI](https://research.nvidia.com/labs/lpr/slm-agents/). Additionally, the [increasing power demands of AI](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report) motivates a turn towards more energy-efficient solutions (such as taking advantage of local computation). And all that is before you even consider how the companies serving large models over the network are [using your information](https://hai.stanford.edu/news/be-careful-what-you-tell-your-ai-chatbot), and to what degree you are comfortable allowing these companies access to your AI conversations and corresponding private data.

That being said, local inference, and especially inference in the browser, is not going to solve all these problems immediately. WebGPU implementations are still maturing, and we ran into issues in the WebGPU implementations of every major browser ([1](https://github.com/gfx-rs/wgpu/issues/8896), [2](https://bugs.webkit.org/show_bug.cgi?id=311598), [3](https://issues.chromium.org/issues/504928024), [4](https://github.com/gpuweb/gpuweb/issues/5315)) while implementing WebGPU support for llama.cpp. Local models, especially in the sizes supported by mobile browsers, are not as capable as users might want, and browsers' same-origin policy means that if a model has been downloaded on one site, it will need to be downloaded again to be used on a different site. Nevertheless, we're still early, and the hope is that implementations like this one lead to demos and use-cases that can help influence the direction of local inference and browser support.

### Disclaimer: Walking in the Footsteps of Others

Before we go any further into the llama.cpp x WebGPU integration, we'd first like to make clear that we are not the first LLM inference runtime for WebGPU. [ONNX Runtime](https://onnxruntime.ai/) was one of the first machine learning frameworks to add web support, and its newer WebGPU backend helps power [Transformers.js](https://huggingface.co/docs/transformers.js/index). [WebLLM](https://webllm.mlc.ai/) brings the power of [TVM](https://tvm.apache.org/) to the browser, efficiently compiling models into deployable formats. In addition, llama.cpp was already capable of being run through the browser, albeit without WebGPU support, using the pre-existing [wllama](https://github.com/ngxson/wllama) library. We are grateful to everyone involved in these efforts, as well as many others that we probably missed, for providing comparisons and starting points, and we're looking forward to seeing the ecosystem around WebGPU and browser-based machine learning continue to evolve.

![Llamas surfing on the web](./llamas-surfing.png "Appropriately, Llamas on the Web was primarily developed at UC Santa Cruz, although the llamas here would probably want a wetsuit to go surfing in Santa Cruz.")

## Llamas on the Web

:::rewrite
Now let's get into the details of the WebGPU x llama.cpp implementation! First, as you've already seen in the demo above, you can now accelerate almost any model supported by llama.cpp using WebGPU, if your device and browser support it. If you're reading this post, you're likely already aware that llama.cpp is already a popular choice for running local models on consumer hardware. With WebGPU support, that choice becomes even easier, as developers can integrate llama.cpp directly into their websites so that even completely non-technical users can benefit from fast, efficient, and private local models.
:::

:::callout
Try rewriting the paragraph above using a local LLM. And if you're entertained or interested by this post and demos, please share it; we'd love help spreading the word!
:::

### Functionality

As of today, the WebGPU backend supports most recent open-weight models, from transformer-based models like Gemma, Qwen, and Llama, to hybrid models like Granite and LFM2, to even experimental 1-bit models like Bonsai. We also support most llama.cpp weight formats, including 32-bit and 16-bit float types, traditional quantization types like q4\_0 and q8\_0, K-quants, and I-quants, allowing you to pick the right model size for your use-case.

Given that WebGPU is still new, there are no mature, portable, and performant libraries for linear algebra routines in WebGPU, such as NVIDIA's [cuBLAS](https://developer.nvidia.com/cublas). Thus, we wrote the WebGPU backend in llama.cpp from the ground up, developing custom WGSL kernels for matrix multiplication, FlashAttention, and all the other important machine learning operations. We have also worked hard to make sure our implementation is as memory-efficient as possible, avoiding overhead and enabling as powerful a model as possible to run on whatever hardware a user has available.

We also want to note that despite the name, the WebGPU backend is not constrained to the browser. If you are developing a native application that needs wide GPU support (especially if it isn't covered by other llama.cpp [backends](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src)), the WebGPU backend works natively using Google's [Dawn](https://github.com/google/dawn) implementation, directly in C++. In fact, since the WebGPU backend lives within the [ggml](https://ggml.ai/) tensor library, you can actually use it independently of llama.cpp as well. However, we expect that most people will want to use it in the context of llama.cpp, which highlights one of our major advantages—llama.cpp's ecosystem.


Nowadays, when a company releases a new open-weight model, it almost always has its architecture baked into llama.cpp on day one. Additionally, thanks to the great work of companies like [Unsloth](https://unsloth.ai/) and the rest of the community, there are numerous model quantizations which allow users to pick the variant with the best capabilities and size for any given hardware. The WebGPU backend inherits this support automatically, meaning that you should be able to deploy new models as soon as they become available.

While you're welcome to roll your own frontend for llama.cpp, we have also integrated WebGPU support into upstream wllama, which is available as an [NPM package](https://www.npmjs.com/package/@wllama/wllama)[^wllama]. Wllama runs llama.cpp within a Web Worker so that inference never blocks the UI-thread, and includes features like automatic model caching. In our experiments, Chromium-based browsers currently provide the fastest WebGPU implementation, but the package should support all major browsers and we know they are also working hard on their WebGPU support.

### Performance

Of course, the main benefit of using WebGPU is its performance. And since seeing is believing, we invite you to run a benchmark yourself below, which compares CPU and GPU performance on your machine:

:::benchmark

Hopefully, that benchmark convinced you that WebGPU is completely worth it. And if it didn't, or the CPU ended up being faster, then please reach out and let us know; we're always looking to improve our performance on every system.

Behind the scenes, we've been working hard to optimize the performance of the major bottlenecks in inference, including matrix and matrix-vector multiplication. One of the most exciting and challenging parts about WebGPU is writing shaders and finding parameters that work well across diverse GPUs, from high-end desktop setups to the smallest mobile phones. We have some very interesting ongoing work in this domain, so stay tuned for more information on it!

Also, if you'd like to contribute data from your own machine or are curious to see performance numbers across other models and GPUs, please visit our [benchmarking site](https://abhijitramesh-webgpu-bench.static.hf.space/index.html) and check out the results.

### Future Work and Technical Report

Obviously there are a ton of juicy technical details that we didn't want to weigh down this release blog post with. If you're interested, we've released a technical report [here](https://arxiv.org/abs/2605.20706), which discusses the design decisions that went into our implementation and reports performance numbers across different browsers, operating systems, models, etc.

With this initial release, we're just getting started. We're already starting to see more contributions to the WebGPU backend in llama.cpp, from improving the performance and portability of different operations, to support for multi-modal vision-language models, to initial kernel fusion implementations. We will also continue to work with browser vendors to find bugs in their implementations and find ways to increase our performance across many devices. If you're interested in helping out or have ideas to help improve the llama.cpp WebGPU backend, please reach out or start a discussion in the [llama.cpp](https://github.com/ggml-org/llama.cpp) repository.

### More Demos

We're interested in seeing how people start using browser-based inference. For a more full-featured chat interface using wllama, visit [this demo](https://reeselevine.github.io/wllama/). And if you make your own, feel free to let us know so we can add it here!

### Acknowledgements

This release wouldn't be possible without the help of many talented people. [Reese Levine](https://reeselevine.github.io) leads the implementation of Llamas on the Web as part of a group at UC Santa Cruz working under [Tyler Sorensen](https://tyler-utah.github.io/) that includes [Rithik Sharma](https://sharmarithik.github.io/rithiksharma/), [Nikhil Jain](https://nikhiljain17.github.io/), [Abhijit Ramesh](https://abhijitramesh.me/), [Zheyuan Chen](https://arbersephirotheca.github.io/), [Neha Abbas](https://www.linkedin.com/in/neha-a-827406275/), and [James Contini](https://jamescontini.com/).

Several outside contributors have also made significant contributions to the WebGPU llama.cpp backend, and we want to especially thank [Masashi Yoshimura](https://github.com/yomaytk) and [Chen Yuan](https://github.com/Constannnnnt) for their work. We'd also like to thank [Georgi Gerganov](https://ggerganov.com/), [Sigbjørn Skjæret](https://github.com/cisc), [Xuan-Son Nguyen](https://ngxson.com/), and the other llama.cpp maintainers for their support of this effort and making this integration possible.

### A Note on the Model Used in This Post

The default model used here is Gemma-3, specifically a 270M parameter Unsloth Q4\_K\_M quantized version. The reason we chose this model is that it is small enough to be loaded on many mobile phones and downloads fairly quickly. However, we had to tailor the demos here to things that come across reasonably well with such a small model. If you are on a machine with a larger WebGPU budget, or looked a the advanced usage section in the model loader, you might have noticed that you could use a Q4\_K version of Qwen 3.5 2B, or choose any model from a Hugging Face repository. Generally, larger and more recent models will feel more capable, at the expense of longer download times and less portability.

[^wllama]: WebGPU support is now available in upstream `@wllama/wllama`. On browsers without JSPI or 64-bit WebAssembly memory support, upstream wllama can fall back to a compat build, including Safari support, though with lower performance.
