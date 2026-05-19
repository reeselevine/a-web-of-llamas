import { memo, useEffect, useRef, useState } from 'react';
import {
  isValidGgufFile,
  ModelManager,
  Wllama,
} from '@reeselevine/wllama-webgpu';
import {
  DEFAULT_DEMO_MODEL_ID,
  DEMO_MODELS,
  PROMPT_OPTIONS,
  WLLAMA_CONFIG_PATHS,
} from './config';
import { formatChat, type ChatMessage } from './chat';
import {
  type BlogNode,
  blogPost,
  buildPromptSource,
  renderInlineMarkdown,
} from './blogpost';

const modelManager = new ModelManager();
const SPLIT_GGUF_REGEX = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/;

const toHumanReadableSize = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
};

const isIOSBrowser = (): boolean => {
  const ua = navigator.userAgent;
  return ua.includes('iPhone') || ua.includes('iPad');
};

const hasWebGPUSupport = (): boolean => {
  return 'gpu' in navigator;
};

const getWebGPUMemoryBudget = async (): Promise<number | undefined> => {
  const gpuNavigator = navigator as Navigator & {
    gpu?: {
      requestAdapter(): Promise<{ limits?: { maxBufferSize?: number } } | null>;
    };
  };

  if (!gpuNavigator.gpu) {
    return undefined;
  }

  const adapter = await gpuNavigator.gpu.requestAdapter();
  const maxBufferSize = adapter?.limits?.maxBufferSize;
  if (!maxBufferSize) {
    return undefined;
  }

  const iosLimit = 512 * 1024 * 1024;
  return isIOSBrowser() ? Math.min(maxBufferSize, iosLimit) : maxBufferSize;
};

const parseSplitFile = (file: string) => {
  const match = file.match(SPLIT_GGUF_REGEX);
  if (!match) {
    return null;
  }

  return {
    current: Number(match[2]),
    total: Number(match[3]),
  };
};

const getSelectableGgufFiles = (files: string[]) =>
  files
    .filter((file) => {
      const split = parseSplitFile(file);
      return !split || split.current === 1;
    })
    .sort((a, b) => a.localeCompare(b));

const getGgufOptionLabel = (file: string) => {
  const split = parseSplitFile(file);
  if (!split) {
    return file;
  }

  return `${file} (${split.total} shards)`;
};

type ActiveModel = {
  id: string;
  name: string;
  modelUrl: string;
  sizeBytes?: number;
};

type PromptSelection = (typeof PROMPT_OPTIONS)[number]['id'] | 'manual';
type RewriteStyle = 'pop song' | 'poem' | 'shakespearian sonnet';
type RewriteState = {
  isLoading: boolean;
  style?: RewriteStyle;
  text?: string;
  error?: string;
};
type BenchmarkMetrics = {
  elapsedMs: number;
  tokens: number;
  tokensPerSecond: number;
};
type BenchmarkBackend = 'cpu' | 'webgpu';
type BenchmarkSelection = 'cpu' | 'webgpu' | 'both';
type BenchmarkRunResult = {
  backend: BenchmarkBackend;
  backendLabel: string;
  threadLabel?: string;
  completed: boolean;
  repetitions: number;
  promptTokens: number;
  prefill: BenchmarkMetrics;
  decode: BenchmarkMetrics;
};
type BenchmarkResult = {
  runs: BenchmarkRunResult[];
  warning?: string;
};
type ActiveBenchmarkMetric = {
  backend: BenchmarkBackend;
  metric: 'prefill' | 'decode';
};
type LoadedModelSnapshot = {
  id: string;
  name: string;
  modelUrl: string;
  backend: BenchmarkBackend;
  nCtx: number;
  nBatch: number;
};

const DEFAULT_MANUAL_PROMPT =
  'Write a paragraph explaining the benefits of running local LLMs, focusing on privacy and control over data.';
const BENCHMARK_PROMPT_TOKEN_COUNT = 512;
const BENCHMARK_DECODE_TOKEN_COUNT = 64;
const BENCHMARK_REPETITIONS = 1;
const BENCHMARK_WARMUP_PREFILL_TOKENS = 2;
const BENCHMARK_WARMUP_DECODE_TOKENS = 1;
const MAIN_STREAM_COMMIT_INTERVAL_MS = 150;
const REWRITE_STREAM_COMMIT_INTERVAL_MS = 150;

const toTokensPerSecond = (tokens: number, elapsedMs: number) =>
  elapsedMs > 0 ? (tokens * 1000) / elapsedMs : 0;

const formatBenchmarkValue = (value: number) =>
  Number.isFinite(value) ? value.toFixed(value >= 100 ? 0 : 1) : '0.0';

const getBenchmarkBackendLabel = (backend: BenchmarkBackend) =>
  backend === 'webgpu' ? 'WebGPU' : 'CPU';

const getBenchmarkThreadLabel = (instance: Wllama) =>
  instance.isMultithread() ? 'multi-thread' : 'single-thread';

const buildRewriteMessages = (
  paragraphText: string,
  style: RewriteStyle
): ChatMessage[] => {
  return [
    {
      role: 'user',
      content: `${paragraphText}\n\Rewrite this as a ${style}.`,
    },
  ];
};

const buildPromptInput = (
  promptId: PromptSelection,
  manualPrompt: string,
  blogPostText: string
) => {
  if (promptId === 'manual') {
    return manualPrompt;
  }

  if (promptId === 'summarize-blogpost') {
    return `Summarize the following blog post in exactly 3 bullet points.\n\n${blogPostText}`;
  }

  return "Write a function that computes the Nth fibonacci number recursively. Give usage for n=10, but don't try to calculate results manually, and don't explain the function.";
};

const getPromptOptionsForPlatform = (isIOS: boolean) => {
  if (!isIOS) {
    return PROMPT_OPTIONS;
  }

  const summarizeOption = PROMPT_OPTIONS.find(
    (option) => option.id === 'summarize-blogpost'
  );
  const remainingOptions = PROMPT_OPTIONS.filter(
    (option) => option.id !== 'summarize-blogpost'
  );

  return summarizeOption
    ? [...remainingOptions, summarizeOption]
    : remainingOptions;
};

const SUMMARY_SECTION_TITLE = 'Llamas on the Web';
const SUMMARY_SUBSECTION_TITLES = new Set([
  'Functionality',
  'Performance',
  'Future Work and Technical Report',
]);
const IPHONE_SUMMARY_SUBSECTION_TITLES = new Set(['Functionality']);

const getBenchmarkBackends = (
  webgpuSupported: boolean,
  selection: BenchmarkSelection
): BenchmarkBackend[] => {
  if (selection === 'cpu') {
    return ['cpu'];
  }

  if (selection === 'webgpu') {
    return webgpuSupported ? ['webgpu'] : [];
  }

  return webgpuSupported ? ['cpu', 'webgpu'] : ['cpu'];
};

const buildSummaryPromptSource = (nodes: BlogNode[], isIPhone: boolean) => {
  const summaryNodes: BlogNode[] = [];
  let inSummarySection = false;
  let includeCurrentSubsection = true;
  const includedSubsections = isIPhone
    ? IPHONE_SUMMARY_SUBSECTION_TITLES
    : SUMMARY_SUBSECTION_TITLES;

  for (const node of nodes) {
    if (node.type === 'heading' && node.level === 2) {
      if (node.text === SUMMARY_SECTION_TITLE) {
        inSummarySection = true;
        includeCurrentSubsection = true;
        summaryNodes.push(node);
        continue;
      }

      if (inSummarySection) {
        break;
      }
    }

    if (!inSummarySection) {
      continue;
    }

    if (node.type === 'heading' && node.level === 3) {
      includeCurrentSubsection = includedSubsections.has(node.text);
      if (includeCurrentSubsection) {
        summaryNodes.push(node);
      }
      continue;
    }

    if (includeCurrentSubsection) {
      summaryNodes.push(node);
    }
  }

  return buildPromptSource(summaryNodes);
};

const { meta: BLOG_META, nodes: BLOG_NODES, footnotes: BLOG_FOOTNOTES } =
  blogPost;
const FOOTNOTE_NUMBERS_BY_ID = new Map(
  BLOG_FOOTNOTES.map((footnote) => [footnote.id, footnote.number] as const)
);
const FIRST_HEADING_INDEX = BLOG_NODES.findIndex(
  (node) => node.type === 'heading'
);
const INTRO_NODES =
  FIRST_HEADING_INDEX === -1
    ? BLOG_NODES
    : BLOG_NODES.slice(0, FIRST_HEADING_INDEX);
const ARTICLE_NODES =
  FIRST_HEADING_INDEX === -1 ? [] : BLOG_NODES.slice(FIRST_HEADING_INDEX);

const StaticArticleNode = memo(function StaticArticleNode({
  node,
  index,
}: {
  node: BlogNode;
  index: number;
}) {
  if (node.type === 'heading') {
    return (
      <section
        key={`heading-${index}`}
        className={`article-block ${
          node.level === 2 ? 'article-heading' : 'article-subheading'
        }`}
      >
        {node.level === 2 ? <h2>{node.text}</h2> : <h3>{node.text}</h3>}
      </section>
    );
  }

  if (node.type === 'paragraph') {
    return (
      <section
        key={`paragraph-${index}`}
        className="article-block article-paragraph"
      >
        <p>{renderInlineMarkdown(node.text, FOOTNOTE_NUMBERS_BY_ID)}</p>
      </section>
    );
  }

  if (node.type === 'callout') {
    return (
      <section key={`callout-${index}`} className="article-block article-callout">
        <div className="callout-box">
          <p>{renderInlineMarkdown(node.text, FOOTNOTE_NUMBERS_BY_ID)}</p>
        </div>
      </section>
    );
  }

  if (node.type !== 'links') {
    return null;
  }

  return (
    <section key={`links-${index}`} className="article-block article-links">
      <div className="links-box">
        {node.items.map((item, itemIndex) => (
          <p key={itemIndex}>{renderInlineMarkdown(item, FOOTNOTE_NUMBERS_BY_ID)}</p>
        ))}
      </div>
    </section>
  );
});

function App() {
  const isIPhone = isIOSBrowser();
  const promptOptions = getPromptOptionsForPlatform(isIPhone);
  const defaultContextLength = isIPhone ? 1024 : 2048;
  const [selectedModelId, setSelectedModelId] = useState(DEFAULT_DEMO_MODEL_ID);
  const [contextLength, setContextLength] = useState(defaultContextLength);
  const [contextLengthInput, setContextLengthInput] = useState(
    String(defaultContextLength)
  );
  const [maxOutputTokens, setMaxOutputTokens] = useState(1024);
  const [maxOutputTokensInput, setMaxOutputTokensInput] = useState('1024');
  const [selectedPromptId, setSelectedPromptId] =
    useState<PromptSelection>('manual');
  const [manualPrompt, setManualPrompt] = useState(DEFAULT_MANUAL_PROMPT);
  const [isShowingPresetPrompt, setIsShowingPresetPrompt] = useState(false);
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState(
    'Load Gemma 3 270M IT to enable the demo.'
  );
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRunningBenchmark, setIsRunningBenchmark] = useState(false);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [runtimeSummary, setRuntimeSummary] = useState<string>('Not loaded');
  const [hasChatTemplate, setHasChatTemplate] = useState<boolean | null>(null);
  const [cachedModelUrls, setCachedModelUrls] = useState<string[]>([]);
  const [cacheSizes, setCacheSizes] = useState<Record<string, number>>({});
  const [isRefreshingCache, setIsRefreshingCache] = useState(false);
  const [webgpuMemoryBudget, setWebgpuMemoryBudget] = useState<
    number | undefined
  >();
  const [customRepo, setCustomRepo] = useState('');
  const [customFile, setCustomFile] = useState('');
  const [customFiles, setCustomFiles] = useState<string[]>([]);
  const [customError, setCustomError] = useState('');
  const [customModelUrl, setCustomModelUrl] = useState<string | null>(null);
  const [customModelName, setCustomModelName] = useState('Custom model');
  const [rewriteOutputs, setRewriteOutputs] = useState<
    Record<string, RewriteState>
  >({});
  const [rewriteSelections, setRewriteSelections] = useState<
    Record<string, RewriteStyle>
  >({});
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(
    null
  );
  const [benchmarkError, setBenchmarkError] = useState('');
  const [benchmarkSelection, setBenchmarkSelection] =
    useState<BenchmarkSelection>('both');
  const [activeBenchmarkMetric, setActiveBenchmarkMetric] =
    useState<ActiveBenchmarkMetric | null>(null);
  const wllamaRef = useRef<Wllama | null>(null);

  useEffect(() => {
    setContextLengthInput(String(contextLength));
  }, [contextLength]);

  useEffect(() => {
    setMaxOutputTokensInput(String(maxOutputTokens));
  }, [maxOutputTokens]);

  const selectedModel =
    DEMO_MODELS.find((model) => model.id === selectedModelId) ??
    DEMO_MODELS.find((model) => model.id === DEFAULT_DEMO_MODEL_ID) ??
    DEMO_MODELS[0];
  const qwenModel = DEMO_MODELS.find((model) => model.id === 'qwen3-5-2b');
  const activeModel: ActiveModel =
    selectedModelId === 'custom' && customModelUrl
      ? {
          id: 'custom',
          name: customModelName,
          modelUrl: customModelUrl,
        }
      : selectedModel;
  const effectiveWebGPUMemoryBudget = webgpuMemoryBudget
    ? Math.floor(webgpuMemoryBudget * 0.8)
    : undefined;
  const qwenBlockedByBudget = !!(
    qwenModel?.sizeBytes &&
    effectiveWebGPUMemoryBudget &&
    qwenModel.sizeBytes > effectiveWebGPUMemoryBudget
  );
  const promptHasContent =
    selectedPromptId === 'manual' ? manualPrompt.trim().length > 0 : true;
  const currentPromptPreview =
    selectedPromptId === 'summarize-blogpost'
      ? 'Summarize the following blog post in exactly 3 bullet points.\n\n<blog text>'
      : buildPromptInput(selectedPromptId, manualPrompt, '');
  const isBusy = isLoadingModel || isGenerating || isRunningBenchmark;

  useEffect(() => {
    refreshCache().catch(console.error);

    let cancelled = false;
    getWebGPUMemoryBudget()
      .then((budget) => {
        if (!cancelled) {
          setWebgpuMemoryBudget(budget);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWebgpuMemoryBudget(undefined);
        }
      });

    return () => {
      cancelled = true;
      const instance = wllamaRef.current;
      if (instance) {
        instance.exit().catch(console.error);
        wllamaRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(async () => {
      if (customRepo.trim().length < 2) {
        setCustomFiles([]);
        setCustomFile('');
        setCustomError('');
        return;
      }

      try {
        const response = await fetch(
          `https://huggingface.co/api/models/${customRepo.trim()}`,
          { signal: controller.signal }
        );
        const data: { siblings?: { rfilename: string }[] } =
          await response.json();

        if (!data.siblings) {
          setCustomFiles([]);
          setCustomFile('');
          setCustomError('No model found, or the repo is private.');
          return;
        }

        const selectableFiles = getSelectableGgufFiles(
          data.siblings
            .map((entry) => entry.rfilename)
            .filter((file) => isValidGgufFile(file))
        );

        setCustomFiles(selectableFiles);
        setCustomError('');
        setCustomFile((currentFile) =>
          selectableFiles.includes(currentFile) ? currentFile : ''
        );
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          setCustomFiles([]);
          setCustomFile('');
          setCustomError(
            error instanceof Error ? error.message : 'Unknown error'
          );
        }
      }
    }, 500);

    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [customRepo]);

  const refreshCache = async () => {
    setIsRefreshingCache(true);
    try {
      const models = await modelManager.getModels();
      setCachedModelUrls(models.map((model) => model.url));
      setCacheSizes(
        Object.fromEntries(models.map((model) => [model.url, model.size]))
      );
    } finally {
      setIsRefreshingCache(false);
    }
  };

  const unloadModel = async () => {
    const instance = wllamaRef.current;
    if (!instance) return;
    setStatus('Unloading model...');
    await instance.exit();
    wllamaRef.current = null;
    setLoadedModelId(null);
    setRuntimeSummary('Not loaded');
    setHasChatTemplate(null);
    setBenchmarkResult(null);
    setBenchmarkError('');
    setDownloadProgress(null);
    setOutput('');
    setStatus('Model unloaded.');
    await refreshCache();
  };

  const clearModelContext = async (instance: Wllama) => {
    try {
      await instance.kvClear();
    } catch (error) {
      console.error('Failed to clear model context.', error);
    }
  };

  const getCurrentLoadedBackend = (instance: Wllama): BenchmarkBackend =>
    instance.usingWebGPU() ? 'webgpu' : 'cpu';

  const ensureFreshInstance = async (backend: BenchmarkBackend = 'webgpu') => {
    if (wllamaRef.current) {
      await wllamaRef.current.exit();
    }
    const instance = new Wllama(WLLAMA_CONFIG_PATHS, {
      backend,
    });
    wllamaRef.current = instance;
    return instance;
  };

  const loadModelForRuntime = async (
    model: ActiveModel | LoadedModelSnapshot,
    backend: BenchmarkBackend,
    config: {
      nCtx: number;
      nBatch: number;
    }
  ) => {
    const instance = await ensureFreshInstance(backend);
    await instance.loadModelFromUrl(model.modelUrl, {
      n_ctx: config.nCtx,
      n_batch: config.nBatch,
    });
    return instance;
  };

  const snapshotLoadedModel = (): LoadedModelSnapshot | null => {
    const instance = wllamaRef.current;
    if (!instance || !loadedModelId) {
      return null;
    }

    const loadedContextInfo = instance.getLoadedContextInfo();
    const loadedModel =
      DEMO_MODELS.find((model) => model.id === loadedModelId) ??
      (loadedModelId === 'custom' && customModelUrl
        ? {
            id: 'custom',
            name: customModelName,
            modelUrl: customModelUrl,
          }
        : null);

    if (!loadedModel) {
      return null;
    }

    return {
      id: loadedModel.id,
      name: loadedModel.name,
      modelUrl: loadedModel.modelUrl,
      backend: getCurrentLoadedBackend(instance),
      nCtx: loadedContextInfo.n_ctx,
      nBatch: loadedContextInfo.n_batch,
    };
  };

  const applyLoadedModelState = (
    modelId: string,
    instance: Wllama,
    statusMessage?: string
  ) => {
    const usingWebGPU = instance.usingWebGPU();
    const isMultithread = instance.isMultithread();
    const contextInfo = instance.getLoadedContextInfo();

    setLoadedModelId(modelId);
    setHasChatTemplate(!!instance.getChatTemplate());
    setRuntimeSummary(
      `${usingWebGPU ? 'WebGPU' : 'CPU'} • ${isMultithread ? 'multithread' : 'single-thread'} • ctx ${contextInfo.n_ctx}`
    );
    setDownloadProgress(1);

    if (statusMessage) {
      setStatus(statusMessage);
    }
  };

  const clearLoadedModelState = () => {
    setLoadedModelId(null);
    setRuntimeSummary('Not loaded');
    setHasChatTemplate(null);
    setDownloadProgress(null);
  };

  const loadSelectedModel = async () => {
    setIsLoadingModel(true);
    setOutput('');
    setBenchmarkResult(null);
    setBenchmarkError('');
    setDownloadProgress(0);
    const webgpuSupported = hasWebGPUSupport();
    setStatus(
      webgpuSupported
        ? `Loading ${activeModel.name}...`
        : `WebGPU is not supported in this browser. Loading ${activeModel.name} with CPU fallback...`
    );
    try {
      const instance = await ensureFreshInstance();
      await instance.loadModelFromUrl(activeModel.modelUrl, {
        n_ctx: contextLength,
        n_batch: 256,
        progressCallback: ({ loaded, total }) => {
          setDownloadProgress(total > 0 ? loaded / total : 0);
        },
      });
      applyLoadedModelState(
        activeModel.id,
        instance,
        instance.usingWebGPU()
          ? `${activeModel.name} is ready.`
          : `${activeModel.name} is ready, but WebGPU is unavailable so it is running on CPU.`
      );
      await refreshCache();
    } catch (error) {
      console.error(error);
      clearLoadedModelState();
      setStatus(
        `Failed to load ${activeModel.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    } finally {
      setIsLoadingModel(false);
    }
  };

  const runPrompt = async () => {
    const instance = wllamaRef.current;
    if (!instance || loadedModelId !== activeModel.id) {
      setStatus('Load the selected model before generating.');
      return;
    }
    setIsGenerating(true);
    setOutput('');
    setStatus('Generating output...');
    try {
      const promptInput = buildPromptInput(
        selectedPromptId,
        manualPrompt,
        buildSummaryPromptSource(BLOG_NODES, isIPhone)
      );
      const formattedPrompt = await formatChat(instance, [
        {
          role: 'user',
          content: promptInput,
        },
      ]);
      console.info('[prompt] rawPrompt', {
        selectedPromptId,
        currentPrompt: promptInput,
      });
      let streamedText = '';
      let lastCommittedText = '';
      let lastCommitTime = 0;
      const result = await instance.createCompletion(formattedPrompt, {
        nPredict: maxOutputTokens,
        sampling: {
          temp: 0.2,
          top_k: 40,
          top_p: 0.9,
        },
        onNewToken(_token, _piece, currentText) {
          streamedText = currentText;
          const now = performance.now();
          if (
            currentText !== lastCommittedText &&
            now - lastCommitTime >= MAIN_STREAM_COMMIT_INTERVAL_MS
          ) {
            lastCommittedText = currentText;
            lastCommitTime = now;
            setOutput(currentText);
          }
        },
      });
      setOutput(result.trim() || streamedText || result);
      setStatus('Generation complete.');
    } catch (error) {
      console.error(error);
      setStatus(
        `Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    } finally {
      await clearModelContext(instance);
      setIsGenerating(false);
    }
  };

  const runBenchmark = async () => {
    setIsRunningBenchmark(true);
    setActiveBenchmarkMetric(null);
    setBenchmarkError('');
    setStatus('Running benchmark...');

    const previousLoadedModel = snapshotLoadedModel();
    const webgpuSupported = hasWebGPUSupport();

    try {
      const benchmarkBackends = getBenchmarkBackends(
        webgpuSupported,
        benchmarkSelection
      );
      if (benchmarkBackends.length === 0) {
        throw new Error('WebGPU is not available in this browser.');
      }
      const benchmarkLoadConfig = {
        nCtx: contextLength,
        nBatch: 256,
      };

      setBenchmarkResult({
        runs: benchmarkBackends.map((backend) => ({
          backend,
          backendLabel: getBenchmarkBackendLabel(backend),
          threadLabel: undefined,
          completed: false,
          repetitions: BENCHMARK_REPETITIONS,
          promptTokens: Math.min(
            BENCHMARK_PROMPT_TOKEN_COUNT,
            benchmarkLoadConfig.nBatch
          ),
          prefill: {
            elapsedMs: 0,
            tokens: Math.min(
              BENCHMARK_PROMPT_TOKEN_COUNT,
              benchmarkLoadConfig.nBatch
            ),
            tokensPerSecond: 0,
          },
          decode: {
            elapsedMs: 0,
            tokens: BENCHMARK_DECODE_TOKEN_COUNT,
            tokensPerSecond: 0,
          },
        })),
        warning:
          !webgpuSupported && benchmarkSelection !== 'cpu'
              ? 'WebGPU is not available in this browser, so only the CPU benchmark can be run.'
              : undefined,
      });

      for (const backend of benchmarkBackends) {
        setStatus(
          `Running ${getBenchmarkBackendLabel(backend)} benchmark...`
        );
        const instance = await loadModelForRuntime(
          activeModel,
          backend,
          benchmarkLoadConfig
        );
        const contextInfo = instance.getLoadedContextInfo();
        const prefillTokenCount = Math.min(
          BENCHMARK_PROMPT_TOKEN_COUNT,
          contextInfo.n_batch
        );

        await instance._testBenchmark(
          'pp',
          Math.min(BENCHMARK_WARMUP_PREFILL_TOKENS, prefillTokenCount)
        );
        await instance._testBenchmark('tg', BENCHMARK_WARMUP_DECODE_TOKENS);

        setActiveBenchmarkMetric({
          backend,
          metric: 'prefill',
        });
        const prefillResult = await instance._testBenchmark(
          'pp',
          prefillTokenCount
        );
        setActiveBenchmarkMetric({
          backend,
          metric: 'decode',
        });
        const decodeResult = await instance._testBenchmark(
          'tg',
          BENCHMARK_DECODE_TOKEN_COUNT
        );

        const completedRun: BenchmarkRunResult = {
          backend,
          backendLabel: getBenchmarkBackendLabel(backend),
          threadLabel: getBenchmarkThreadLabel(instance),
          completed: true,
          repetitions: BENCHMARK_REPETITIONS,
          promptTokens: prefillTokenCount,
          prefill: {
            elapsedMs: prefillResult.t_ms,
            tokens: prefillTokenCount,
            tokensPerSecond: toTokensPerSecond(
              prefillTokenCount,
              prefillResult.t_ms
            ),
          },
          decode: {
            elapsedMs: decodeResult.t_ms,
            tokens: BENCHMARK_DECODE_TOKEN_COUNT,
            tokensPerSecond: toTokensPerSecond(
              BENCHMARK_DECODE_TOKEN_COUNT,
              decodeResult.t_ms
            ),
          },
        };

        setBenchmarkResult((current) =>
          current
            ? {
                ...current,
                runs: current.runs.map((run) =>
                  run.backend === backend ? completedRun : run
                ),
              }
            : current
        );
        setActiveBenchmarkMetric(null);
      }
      setStatus('Benchmark complete.');
    } catch (error) {
      console.error(error);
      const message =
        error instanceof Error ? error.message : 'Unknown benchmark error';
      setBenchmarkError(message);
      setStatus(`Benchmark failed: ${message}`);
    } finally {
      try {
        if (previousLoadedModel) {
          setStatus(`Restoring ${previousLoadedModel.name}...`);
          const restoredInstance = await loadModelForRuntime(
            previousLoadedModel,
            previousLoadedModel.backend,
            {
              nCtx: previousLoadedModel.nCtx,
              nBatch: previousLoadedModel.nBatch,
            }
          );
          applyLoadedModelState(previousLoadedModel.id, restoredInstance);
        } else {
          const instance = wllamaRef.current;
          if (instance) {
            await instance.exit();
            wllamaRef.current = null;
          }
          clearLoadedModelState();
        }
      } catch (restoreError) {
        console.error(restoreError);
        clearLoadedModelState();
        setStatus(
          `Benchmark finished, but restoring the previous model failed: ${
            restoreError instanceof Error
              ? restoreError.message
              : 'Unknown restore error'
          }`
        );
      }
      setActiveBenchmarkMetric(null);
      setIsRunningBenchmark(false);
    }
  };

  const rewriteParagraph = async (
    nodeId: string,
    paragraphText: string,
    style: RewriteStyle
  ) => {
    const instance = wllamaRef.current;
    if (!instance || loadedModelId !== activeModel.id) {
      setStatus('Load the selected model before generating a rewrite.');
      return;
    }

    setRewriteOutputs((current) => ({
      ...current,
      [nodeId]: {
        ...current[nodeId],
        isLoading: true,
        style,
        text: '',
        error: undefined,
      },
    }));

    try {
      const rewriteMessages = buildRewriteMessages(paragraphText, style);
      const formattedPrompt = await formatChat(instance, rewriteMessages);
      let streamedText = '';
      let lastCommittedText = '';
      let lastCommitTime = 0;
      console.info('[rewrite] rawPrompt', {
        nodeId,
        style,
        messages: rewriteMessages,
      });
      const result = await instance.createCompletion(formattedPrompt, {
        nPredict: maxOutputTokens,
        sampling: {
          temp: 0.3,
          top_k: 40,
          top_p: 0.9,
        },
        onNewToken(_token, _piece, currentText) {
          streamedText = currentText;
          const now = performance.now();
          if (
            currentText !== lastCommittedText &&
            now - lastCommitTime >= REWRITE_STREAM_COMMIT_INTERVAL_MS
          ) {
            lastCommittedText = currentText;
            lastCommitTime = now;
            setRewriteOutputs((current) => ({
              ...current,
              [nodeId]: {
                ...current[nodeId],
                isLoading: true,
                style,
                text: currentText,
                error: undefined,
              },
            }));
          }
        },
      });
      const finalText = (result.trim() || streamedText || result).trim();
      console.info('[rewrite] result', {
        nodeId,
        style,
        result,
        streamedText,
        finalText,
      });

      setRewriteOutputs((current) => ({
        ...current,
        [nodeId]: {
          isLoading: false,
          style,
          text: finalText,
        },
      }));
    } catch (error) {
      console.error(error);
      setRewriteOutputs((current) => ({
        ...current,
        [nodeId]: {
          isLoading: false,
          style,
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      }));
    } finally {
      await clearModelContext(instance);
    }
  };

  const loadedModelUrl = DEMO_MODELS.find(
    (model) => model.id === loadedModelId
  )?.modelUrl ?? (loadedModelId === 'custom' ? customModelUrl ?? undefined : undefined);

  const removeCachedModel = async (modelUrl: string) => {
    setStatus('Removing cached model...');
    try {
      if (loadedModelUrl && modelUrl === loadedModelUrl) {
        await unloadModel();
      } else {
        const models = await modelManager.getModels();
        const model = models.find((entry) => entry.url === modelUrl);
        if (model) {
          await model.remove();
        }
        await refreshCache();
        setStatus('Cached model removed.');
      }
    } catch (error) {
      console.error(error);
      setStatus(
        `Failed to remove cached model: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  };

  const cachedDemoModels = DEMO_MODELS.filter((model) =>
    cachedModelUrls.includes(model.modelUrl)
  );
  const isActiveModelLoaded = loadedModelId === activeModel.id;
  const benchmarkDisplayBackends = getBenchmarkBackends(
    hasWebGPUSupport(),
    benchmarkSelection
  );
  const benchmarkDisplayRuns =
    benchmarkResult?.runs ??
    benchmarkDisplayBackends.map((backend) => ({
        backend,
        backendLabel: getBenchmarkBackendLabel(backend),
        threadLabel: undefined,
        completed: false,
        repetitions: BENCHMARK_REPETITIONS,
        promptTokens: 256,
        prefill: {
          elapsedMs: 0,
          tokens: 256,
          tokensPerSecond: 0,
        },
        decode: {
          elapsedMs: 0,
          tokens: BENCHMARK_DECODE_TOKEN_COUNT,
          tokensPerSecond: 0,
        },
      }));
  const cpuBenchmarkRun = benchmarkDisplayRuns.find(
    (run) => run.backend === 'cpu'
  );
  const webgpuBenchmarkRun = benchmarkDisplayRuns.find(
    (run) => run.backend === 'webgpu'
  );
  const prefillSpeedup =
    cpuBenchmarkRun?.completed && webgpuBenchmarkRun?.completed
      ? webgpuBenchmarkRun.prefill.tokensPerSecond /
        cpuBenchmarkRun.prefill.tokensPerSecond
      : null;
  const decodeSpeedup =
    cpuBenchmarkRun?.completed && webgpuBenchmarkRun?.completed
      ? webgpuBenchmarkRun.decode.tokensPerSecond /
        cpuBenchmarkRun.decode.tokensPerSecond
      : null;
  const maxPrefillTokensPerSecond = Math.max(
    cpuBenchmarkRun?.prefill.tokensPerSecond ?? 0,
    webgpuBenchmarkRun?.prefill.tokensPerSecond ?? 0
  );
  const maxDecodeTokensPerSecond = Math.max(
    cpuBenchmarkRun?.decode.tokensPerSecond ?? 0,
    webgpuBenchmarkRun?.decode.tokensPerSecond ?? 0
  );
  const benchmarkRunLabel = (run: BenchmarkRunResult) =>
    run.backend === 'cpu'
      ? run.threadLabel
        ? `CPU (${run.threadLabel})`
        : 'CPU'
      : 'WebGPU';

  const renderInlineLoadModelButton = (label = 'Load model') => (
    <button
      type="button"
      className="inline-pill-button"
      onClick={() => {
        loadSelectedModel().catch(console.error);
      }}
      disabled={isBusy || isActiveModelLoaded}
    >
      {isLoadingModel ? 'Loading...' : label}
    </button>
  );

  const formatBenchmarkMetricText = (
    run: BenchmarkRunResult | undefined,
    metric: 'prefill' | 'decode'
  ) => {
    const isActiveMetric =
      activeBenchmarkMetric !== null &&
      activeBenchmarkMetric.backend === run?.backend &&
      activeBenchmarkMetric.metric === metric;

    return run && run.completed
      ? `${formatBenchmarkValue(run[metric].tokensPerSecond)} tok/s`
      : isActiveMetric
        ? 'Running...'
        : 'Waiting...';
  };

  const renderBenchmarkCard = () => (
    <div className="benchmark-card">
      <div className="benchmark-header">
        <div>
          <span className="runtime-label">Benchmark</span>
          <h4>Prefill + Decode</h4>
          <p className="benchmark-model-name">{activeModel.name}</p>
        </div>
        <button
          type="button"
          className="inline-pill-button benchmark-button"
          onClick={() => {
            runBenchmark().catch(console.error);
          }}
          disabled={isBusy}
        >
          {isRunningBenchmark ? 'Running...' : 'Run Benchmark'}
        </button>
      </div>
      <p className="advanced-note benchmark-note">
        This set of benchmarks runs the active model on the CPU and, when available,
        through WebGPU. Each benchmark gets a warmup pass, then one measured run.
        Prefill measures the speed at which the model can process the prompt and prepare for generation, while
        decode measures the speed at which the model can generate tokens.
      </p>
      <p className="advanced-warning benchmark-note">
        On smaller devices, especially iPhones, WebGPU benchmark runs may raise
        memory usage enough to crash the tab. Since this page is currently
        hosted on GitHub Pages, WebAssembly threads{' '}
        <a
          href="https://github.com/orgs/community/discussions/13309"
          target="_blank"
          rel="noreferrer"
        >
          cannot be enabled
        </a>
        . Multi-threading CPU performance is generally better, but still not as
        fast as WebGPU on machines we've tested.
      </p>
      <div className="benchmark-controls">
        <label className="field">
          <span>Benchmark mode</span>
          <select
            value={benchmarkSelection}
            onChange={(event) =>
              setBenchmarkSelection(event.target.value as BenchmarkSelection)
            }
            disabled={isBusy}
          >
            <option value="both">CPU + WebGPU</option>
            <option value="cpu">CPU only</option>
            <option value="webgpu" disabled={!hasWebGPUSupport()}>
              WebGPU only
            </option>
          </select>
        </label>
      </div>
      <>
          <div className="benchmark-chart">
            <div className="benchmark-chart-group">
              <div className="benchmark-chart-header">
                <span className="runtime-label">Prefill</span>
                <span>
                  {benchmarkDisplayRuns[0]?.prefill.tokens ?? 256} tokens
                </span>
              </div>
              {benchmarkDisplayRuns.map((run) => (
                <div className="benchmark-bar-row">
                  <span>{benchmarkRunLabel(run)}</span>
                  <div className="benchmark-bar-track" aria-hidden="true">
                    <div
                      className={`benchmark-bar-fill ${
                        run.backend === 'cpu' ? 'cpu-fill' : 'gpu-fill'
                      }`}
                      style={{
                        width: `${
                          maxPrefillTokensPerSecond > 0
                            ? (run.prefill.tokensPerSecond /
                                maxPrefillTokensPerSecond) *
                              100
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                  <strong>{formatBenchmarkMetricText(run, 'prefill')}</strong>
                </div>
              ))}
            </div>
            <div className="benchmark-chart-group">
              <div className="benchmark-chart-header">
                <span className="runtime-label">Decode</span>
                <span>
                  {benchmarkDisplayRuns[0]?.decode.tokens ??
                    BENCHMARK_DECODE_TOKEN_COUNT}{' '}
                  tokens
                </span>
              </div>
              {benchmarkDisplayRuns.map((run) => (
                <div className="benchmark-bar-row">
                  <span>{benchmarkRunLabel(run)}</span>
                  <div className="benchmark-bar-track" aria-hidden="true">
                    <div
                      className={`benchmark-bar-fill ${
                        run.backend === 'cpu' ? 'cpu-fill' : 'gpu-fill'
                      }`}
                      style={{
                        width: `${
                          maxDecodeTokensPerSecond > 0
                            ? (run.decode.tokensPerSecond /
                                maxDecodeTokensPerSecond) *
                              100
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                  <strong>{formatBenchmarkMetricText(run, 'decode')}</strong>
                </div>
              ))}
            </div>
          </div>
          {prefillSpeedup && decodeSpeedup ? (
            <p className="benchmark-summary">
              WebGPU is {formatBenchmarkValue(prefillSpeedup)}x faster on
              prefill and {formatBenchmarkValue(decodeSpeedup)}x faster on
              decode for this run.
            </p>
          ) : null}
      </>
      {benchmarkResult?.warning ? (
        <p className="advanced-warning">{benchmarkResult.warning}</p>
      ) : null}
      {benchmarkError ? (
        <p className="advanced-warning">{benchmarkError}</p>
      ) : null}
    </div>
  );

  const renderArticleNode = (node: BlogNode, index: number) => {
    if (
      node.type === 'heading' ||
      node.type === 'paragraph' ||
      node.type === 'callout' ||
      node.type === 'links'
    ) {
      return <StaticArticleNode key={`${node.type}-${index}`} node={node} index={index} />;
    }

    if (node.type === 'benchmark') {
      return (
        <section
          key={`benchmark-${index}`}
          className="article-block article-benchmark"
        >
          {renderBenchmarkCard()}
        </section>
      );
    }

    if (node.type === 'rewrite') {
      const rewriteState = rewriteOutputs[node.id];
      const selectedRewriteStyle =
        rewriteSelections[node.id] ?? 'pop song';
      const displayedText = rewriteState?.text || node.text;

      return (
        <section
          key={`rewrite-${node.id}`}
          className="article-block article-rewrite-block"
        >
          <div className="rewrite-copy">
            {rewriteState?.text ? (
              <p className="rewrite-rendered">{displayedText}</p>
            ) : (
              <p>{renderInlineMarkdown(node.text, FOOTNOTE_NUMBERS_BY_ID)}</p>
            )}
          </div>
          <aside className="rewrite-sidebar">
            <span className="runtime-label">Rewrite This Paragraph</span>
            <div className="rewrite-actions">
              <label className="field rewrite-style-field">
                <span>Style</span>
                <select
                  value={selectedRewriteStyle}
                  onChange={(event) =>
                    setRewriteSelections((current) => ({
                      ...current,
                      [node.id]: event.target.value as RewriteStyle,
                    }))
                  }
                  disabled={
                    isBusy || rewriteState?.isLoading
                  }
                >
                  <option value="pop song">Pop song</option>
                  <option value="poem">Poem</option>
                  <option value="shakespearian sonnet">
                    Shakespearian sonnet
                  </option>
                </select>
              </label>
              <div className="rewrite-button-row">
                <button
                  type="button"
                  className="inline-pill-button rewrite-button"
                  onClick={() => {
                    rewriteParagraph(
                      node.id,
                      node.text,
                      selectedRewriteStyle
                    ).catch(console.error);
                  }}
                  disabled={
                    isBusy ||
                    !isActiveModelLoaded ||
                    rewriteState?.isLoading
                  }
                >
                  Rewrite
                </button>
                <button
                  type="button"
                  className="inline-pill-button tertiary-pill-button"
                  onClick={() => {
                    setRewriteOutputs((current) => {
                      const next = { ...current };
                      delete next[node.id];
                      return next;
                    });
                  }}
                  disabled={rewriteState?.isLoading || !rewriteState?.text}
                >
                  Reset
                </button>
              </div>
              {!isActiveModelLoaded ? (
                <div className="rewrite-load-action">
                  {renderInlineLoadModelButton()}
                </div>
              ) : null}
            </div>
            {rewriteState?.isLoading ? (
              <p className="advanced-note">Rewriting...</p>
            ) : null}
            {rewriteState?.error ? (
              <p className="advanced-warning">{rewriteState.error}</p>
            ) : null}
          </aside>
        </section>
      );
    }

    return (
      <section key={`demo-${index}`} className="article-block article-demo">
          <div className="demo-placeholder">
            <p className="section-label">Live Demo</p>
            <p>The first time you run this demo, it will download a small model that will run on your device. After that, the model will be cached for future use (try it in airplane mode or wifi turned off!).</p>
            <p className="advanced-warning">
              This demo may crash or not work correctly on smaller devices such as iPhones due to memory constraints.
            </p>
            <div className="demo-controls">
            <div className="default-model-card">
              <span className="runtime-label">
                {selectedModelId === DEFAULT_DEMO_MODEL_ID
                  ? 'Default model'
                  : 'Current model'}
              </span>
              <h4>{activeModel.name}</h4>
              {loadedModelId === activeModel.id ? (
                <p
                  className={`runtime-support ${runtimeSummary.startsWith('WebGPU') ? 'supported' : 'unsupported'}`}
                >
                  <span aria-hidden="true">
                    {runtimeSummary.startsWith('WebGPU') ? '✓' : '✕'}
                  </span>
                  <span>
                    {runtimeSummary.startsWith('WebGPU')
                      ? 'WebGPU enabled'
                      : 'WebGPU unavailable'}
                  </span>
                </p>
              ) : null}
              {selectedModelId !== DEFAULT_DEMO_MODEL_ID ? (
                <button
                  type="button"
                  className="inline-pill-button tertiary-pill-button"
                  onClick={() => {
                    setSelectedModelId(DEFAULT_DEMO_MODEL_ID);
                    setStatus(`Selected ${DEMO_MODELS[0].name}.`);
                  }}
                  disabled={isBusy}
                >
                  Reset model
                </button>
              ) : null}
            </div>
            <div className="button-row">
              <button
                type="button"
                onClick={loadSelectedModel}
                disabled={isBusy || loadedModelId === activeModel.id}
              >
                {isLoadingModel ? 'Loading...' : 'Load model'}
              </button>
              <button
                type="button"
                className="tertiary-button"
                onClick={() => {
                  unloadModel().catch(console.error);
                }}
                disabled={isBusy || !loadedModelId}
              >
                Unload
              </button>
            </div>
            <div className="progress-block">
              <div className="progress-copy">
                <span className="runtime-label">Download progress</span>
                <span>
                  {downloadProgress === null
                    ? loadedModelId === activeModel.id
                      ? 'Cached and ready'
                      : 'Idle'
                    : `${Math.round(downloadProgress * 100)}%`}
                </span>
              </div>
              <div className="progress-track" aria-hidden="true">
                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(100, Math.round((downloadProgress ?? 0) * 100))
                    )}%`,
                  }}
                />
              </div>
            </div>
            <details className="advanced-usage">
              <summary>Advanced Usage</summary>
              <div className="advanced-usage-panel">
                {qwenModel ? (
                  <div className="advanced-option">
                    <div>
                      <span className="runtime-label">Larger model</span>
                      <h4>{qwenModel.name}</h4>
                      {qwenBlockedByBudget ? (
                        <p className="advanced-warning">
                          Too large for the current WebGPU budget
                          {effectiveWebGPUMemoryBudget
                            ? ` (${toHumanReadableSize(effectiveWebGPUMemoryBudget)})`
                            : ''}
                          .
                        </p>
                      ) : effectiveWebGPUMemoryBudget ? (
                        <p className="advanced-note">
                          Fits within the detected WebGPU budget.
                        </p>
                      ) : (
                        <p className="advanced-note">
                          WebGPU budget unavailable, so compatibility is unknown.
                        </p>
                      )}
                    </div>
                    <button
                      type="button"
                      className="inline-pill-button"
                      onClick={() => {
                        setSelectedModelId(qwenModel.id);
                        setStatus(`Selected ${qwenModel.name}.`);
                      }}
                      disabled={isBusy || qwenBlockedByBudget}
                    >
                      Use {qwenModel.name}
                    </button>
                  </div>
                ) : null}
                <div className="advanced-option advanced-custom-model">
                  <div>
                    <span className="runtime-label">Custom Hugging Face model</span>
                    <h4>Add a GGUF from any public repo</h4>
                    <p className="advanced-note">
                      Enter a repo, then choose from valid single GGUF files or
                      first shards only.
                    </p>
                  </div>
                  <label className="field">
                    <span>HF repo</span>
                    <input
                      type="text"
                      placeholder="username/repo"
                      value={customRepo}
                      onChange={(event) => {
                        setCustomRepo(event.target.value);
                        setCustomModelUrl(null);
                        setCustomModelName('Custom model');
                      }}
                      disabled={isBusy}
                    />
                  </label>
                  <label className="field">
                    <span>GGUF file</span>
                    <select
                      value={customFile}
                      onChange={(event) => setCustomFile(event.target.value)}
                      disabled={isBusy || customRepo.trim().length < 2}
                    >
                      <option value="">Select a model file</option>
                      {customFiles.map((file) => (
                        <option key={file} value={file}>
                          {getGgufOptionLabel(file)}
                        </option>
                      ))}
                    </select>
                  </label>
                  {customFiles.length > 0 ? (
                    <p className="advanced-note">
                      Showing single GGUF files and first shards only.
                    </p>
                  ) : null}
                  {customError ? (
                    <p className="advanced-warning">{customError}</p>
                  ) : null}
                  <button
                    type="button"
                    className="inline-pill-button"
                    onClick={() => {
                      const repo = customRepo.trim();
                      if (!repo || !customFile) return;
                      setCustomModelUrl(
                        `https://huggingface.co/${repo}/resolve/main/${customFile}`
                      );
                      setCustomModelName(`${repo}/${customFile}`);
                      setSelectedModelId('custom');
                      setStatus(`Selected ${repo}/${customFile}.`);
                    }}
                    disabled={
                      isBusy ||
                      customRepo.trim().length < 2 ||
                      customFile.length < 5
                    }
                  >
                    Use custom model
                  </button>
                </div>
                <div className="advanced-option">
                  <div>
                    <span className="runtime-label">Context length</span>
                    <h4>Adjust the prompt window</h4>
                    <p className="advanced-note">
                      Larger values use more memory and may make model loading
                      fail on smaller devices.
                    </p>
                  </div>
                  <label className="field">
                    <span>Context length</span>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={contextLengthInput}
                      onChange={(event) => {
                        const { value } = event.target;
                        setContextLengthInput(value);
                        if (value === '') {
                          return;
                        }

                        const nextValue = Number(value);
                        if (Number.isFinite(nextValue) && nextValue >= 1) {
                          setContextLength(Math.floor(nextValue));
                        }
                      }}
                      onBlur={() => {
                        if (contextLengthInput === '') {
                          setContextLengthInput(String(contextLength));
                        }
                      }}
                      disabled={isBusy}
                    />
                  </label>
                </div>
                <div className="advanced-option">
                  <div>
                    <span className="runtime-label">Max output tokens</span>
                    <h4>Limit generated output length</h4>
                    <p className="advanced-note">
                      Higher limits allow longer responses but take more time to
                      generate.
                    </p>
                  </div>
                  <label className="field">
                    <span>Max output tokens</span>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={maxOutputTokensInput}
                      onChange={(event) => {
                        const { value } = event.target;
                        setMaxOutputTokensInput(value);
                        if (value === '') {
                          return;
                        }

                        const nextValue = Number(value);
                        if (Number.isFinite(nextValue) && nextValue >= 1) {
                          setMaxOutputTokens(Math.floor(nextValue));
                        }
                      }}
                      onBlur={() => {
                        if (maxOutputTokensInput === '') {
                          setMaxOutputTokensInput(String(maxOutputTokens));
                        }
                      }}
                      disabled={isBusy}
                    />
                  </label>
                </div>
              </div>
            </details>
            <details className="advanced-usage cache-disclosure">
              <summary>Cached Models</summary>
              <div className="advanced-usage-panel cache-block">
                <div className="cache-header">
                  <span className="runtime-label">Stored locally</span>
                  <button
                    type="button"
                    className="inline-action"
                    onClick={() => {
                      refreshCache().catch(console.error);
                    }}
                    disabled={isRefreshingCache || isBusy}
                  >
                    {isRefreshingCache ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
                {cachedDemoModels.length === 0 ? (
                  <p className="cache-empty">
                    None of the demo models are cached locally yet.
                  </p>
                ) : (
                  <ul className="cache-list">
                    {cachedDemoModels.map((model) => (
                      <li key={model.id} className="cache-item">
                        <div>
                          <strong>{model.name}</strong>
                          <p>
                            {toHumanReadableSize(cacheSizes[model.modelUrl] ?? 0)}
                          </p>
                        </div>
                        <button
                          type="button"
                          className="inline-action danger-action"
                          onClick={() => {
                            removeCachedModel(model.modelUrl).catch(
                              console.error
                            );
                          }}
                          disabled={isBusy}
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </details>
            <div className="prompt-toolbar">
              <label className="field prompt-selector">
                <span>Prompt</span>
                <select
                  value={selectedPromptId}
                  onChange={(event) => {
                    setSelectedPromptId(event.target.value as PromptSelection);
                    setIsShowingPresetPrompt(false);
                  }}
                  disabled={isBusy}
                >
                  {promptOptions.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                  <option value="manual">Custom prompt</option>
                </select>
              </label>
              <button
                type="button"
                className="inline-pill-button prompt-run-button"
                onClick={() => {
                  runPrompt().catch(console.error);
                }}
                disabled={
                  isBusy ||
                  loadedModelId !== activeModel.id ||
                  !promptHasContent
                }
              >
                {isGenerating ? 'Generating...' : 'Run prompt'}
              </button>
            </div>
            {selectedPromptId !== 'manual' ? (
              <button
                type="button"
                className="inline-action prompt-toggle"
                onClick={() =>
                  setIsShowingPresetPrompt((isShowing) => !isShowing)
                }
                disabled={isBusy}
              >
                {isShowingPresetPrompt ? 'Hide prompt' : 'See prompt'}
              </button>
            ) : null}
            {selectedPromptId !== 'manual' && isShowingPresetPrompt ? (
              <label className="field">
                <span>Prompt</span>
                <textarea value={currentPromptPreview} rows={3} readOnly />
              </label>
            ) : null}
            {selectedPromptId === 'manual' ? (
              <>
                <label className="field">
                  <span>Custom prompt</span>
                  <textarea
                    className={
                      manualPrompt === DEFAULT_MANUAL_PROMPT
                        ? 'default-manual-prompt'
                        : undefined
                    }
                    value={manualPrompt}
                    onChange={(event) => setManualPrompt(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        runPrompt().catch(console.error);
                      }
                    }}
                    rows={3}
                    disabled={isBusy}
                  />
                </label>
                <div className="manual-prompt-warning">
                  Smaller models may hallucinate or generate incorrect results. They are generally better at more structured tasks.
                </div>
              </>
            ) : null}
            <div className="output-text">
              {output || 'Model output will appear here.'}
            </div>
          </div>
        </div>
      </section>
    );
  };

  return (
    <div className="page-shell">
      <header className="blog-header">
        <h1>{BLOG_META.title}</h1>
        {BLOG_META.subtitle ? (
          <p className="blog-subtitle">{BLOG_META.subtitle}</p>
        ) : null}
        <div className="byline">
          {BLOG_META.byline.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </header>

      <main className="blog-layout">
        <article className="blog-article">
          {INTRO_NODES.length > 0 ? (
            <div className="blog-intro">
              {INTRO_NODES.map((node, index) => renderArticleNode(node, index))}
            </div>
          ) : null}

          {ARTICLE_NODES.map((node, index) =>
            renderArticleNode(node, INTRO_NODES.length + index)
          )}
          {BLOG_FOOTNOTES.length > 0 ? (
            <section className="article-block article-footnotes">
              <h3>Footnotes</h3>
              <ol className="footnote-list">
                {BLOG_FOOTNOTES.map((footnote) => (
                  <li key={footnote.id} id={`footnote-${footnote.number}`}>
                    {renderInlineMarkdown(
                      footnote.text,
                      FOOTNOTE_NUMBERS_BY_ID
                    )}
                  </li>
                ))}
              </ol>
            </section>
          ) : null}
        </article>
      </main>
    </div>
  );
}

export default App;
