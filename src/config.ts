import wllamaWasm from '@wllama/wllama/src/wasm/wllama.wasm?url';

export const WLLAMA_CONFIG_PATHS = {
  default: wllamaWasm,
};

export type DemoModel = {
  id: string;
  name: string;
  modelUrl: string;
  sizeBytes?: number;
};

export const DEMO_MODELS: DemoModel[] = [
  {
    id: 'gemma-3-270m',
    name: 'Gemma 3 270M IT',
    sizeBytes: 260046848,
    modelUrl:
      'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf',
  },
  {
    id: 'qwen3-5-2b',
    name: 'Qwen 3.5 2B',
    sizeBytes: 1290000000,
    modelUrl:
      'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf',
  },
];

export const DEFAULT_DEMO_MODEL_ID = 'gemma-3-270m';

export const PROMPT_OPTIONS = [
  {
    id: 'summarize-blogpost',
    label: 'Summarize Llamas on the Web',
  },
  {
    id: 'fib-recursive',
    label: 'Write a Fibonacci function',
  },
] as const;

export const DEFAULT_CHAT_TEMPLATE =
  "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
