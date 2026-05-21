import { Fragment, ReactNode } from 'react';
import rawBlogpost from './content/blogpost.md?raw';

type BlogPostMeta = {
  title: string;
  subtitle?: string;
  byline: string[];
};

export type Footnote = {
  id: string;
  text: string;
  number: number;
};

export type BlogNode =
  | { type: 'heading'; level: 2 | 3; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'image'; alt: string; src: string; caption?: string }
  | { type: 'rewrite'; id: string; text: string }
  | { type: 'callout'; text: string }
  | { type: 'links'; items: string[] }
  | { type: 'demo' }
  | { type: 'benchmark' };

export type BlogPost = {
  meta: BlogPostMeta;
  nodes: BlogNode[];
  footnotes: Footnote[];
};

const FRONTMATTER_KEYS = {
  title: 'title',
  subtitle: 'subtitle',
  byline: 'byline',
} as const;

const requiredMetaKeys = [
  FRONTMATTER_KEYS.title,
  FRONTMATTER_KEYS.byline,
] as const;

const parseFrontmatter = (
  source: string
): { frontmatter: Record<string, string>; body: string } => {
  if (!source.startsWith('---\n')) {
    throw new Error('Blogpost markdown is missing frontmatter.');
  }

  const closingIndex = source.indexOf('\n---\n', 4);
  if (closingIndex === -1) {
    throw new Error('Blogpost markdown frontmatter is not terminated.');
  }

  const frontmatterBlock = source.slice(4, closingIndex);
  const body = source.slice(closingIndex + 5).trim();
  const frontmatter = Object.fromEntries(
    frontmatterBlock
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        const separatorIndex = line.indexOf(':');
        if (separatorIndex === -1) {
          throw new Error(`Invalid frontmatter line: ${line}`);
        }
        const key = line.slice(0, separatorIndex).trim();
        const value = line.slice(separatorIndex + 1).trim();
        return [key, value];
      })
  );

  return { frontmatter, body };
};

const FOOTNOTE_DEFINITION_REGEX = /^\[\^([^\]]+)\]:\s*(.*)$/;
const IMAGE_REGEX = /^!\[([^\]]*)\]\((\S+)(?:\s+"([^"]+)")?\)$/;

const parseNodes = (
  body: string
): { nodes: BlogNode[]; footnotes: Footnote[] } => {
  const lines = body.replace(/\r\n/g, '\n').split('\n');
  const nodes: BlogNode[] = [];
  const footnotes: Footnote[] = [];
  let paragraphLines: string[] = [];
  let rewriteIndex = 0;

  const flushParagraph = () => {
    if (paragraphLines.length === 0) return;
    nodes.push({
      type: 'paragraph',
      text: paragraphLines.join(' ').trim(),
    });
    paragraphLines = [];
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      continue;
    }

    if (trimmed === ':::demo') {
      flushParagraph();
      nodes.push({ type: 'demo' });
      continue;
    }

    if (trimmed === ':::benchmark') {
      flushParagraph();
      nodes.push({ type: 'benchmark' });
      continue;
    }

    if (trimmed === ':::callout') {
      flushParagraph();
      const calloutLines: string[] = [];

      while (i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        i += 1;
        if (nextLine.trim() === ':::') {
          break;
        }
        calloutLines.push(nextLine.trim());
      }

      nodes.push({
        type: 'callout',
        text: calloutLines.join(' ').trim(),
      });
      continue;
    }

    if (trimmed === ':::links') {
      flushParagraph();
      const linkLines: string[] = [];

      while (i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        i += 1;
        if (nextLine.trim() === ':::') {
          break;
        }
        if (nextLine.trim()) {
          linkLines.push(nextLine.trim());
        }
      }

      nodes.push({
        type: 'links',
        items: linkLines,
      });
      continue;
    }

    if (trimmed === ':::rewrite') {
      flushParagraph();
      const rewriteLines: string[] = [];

      while (i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        i += 1;
        if (nextLine.trim() === ':::') {
          break;
        }
        rewriteLines.push(nextLine.trim());
      }

      rewriteIndex += 1;
      nodes.push({
        type: 'rewrite',
        id: `rewrite-${rewriteIndex}`,
        text: rewriteLines.join(' ').trim(),
      });
      continue;
    }

    if (trimmed.startsWith('### ')) {
      flushParagraph();
      nodes.push({ type: 'heading', level: 3, text: trimmed.slice(4).trim() });
      continue;
    }

    if (trimmed.startsWith('## ')) {
      flushParagraph();
      nodes.push({ type: 'heading', level: 2, text: trimmed.slice(3).trim() });
      continue;
    }

    const footnoteMatch = trimmed.match(FOOTNOTE_DEFINITION_REGEX);
    if (footnoteMatch) {
      flushParagraph();
      const footnoteLines = [footnoteMatch[2].trim()];

      while (i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        if (!nextLine.trim()) {
          break;
        }

        if (!/^(?:\t| {2,})/.test(nextLine)) {
          break;
        }

        i += 1;
        footnoteLines.push(nextLine.trim());
      }

      footnotes.push({
        id: footnoteMatch[1].trim(),
        text: footnoteLines.join(' ').trim(),
        number: footnotes.length + 1,
      });
      continue;
    }

    const imageMatch = trimmed.match(IMAGE_REGEX);
    if (imageMatch) {
      flushParagraph();
      nodes.push({
        type: 'image',
        alt: imageMatch[1].trim(),
        src: imageMatch[2].trim(),
        caption: imageMatch[3]?.trim(),
      });
      continue;
    }

    paragraphLines.push(trimmed);
  }

  flushParagraph();
  return { nodes, footnotes };
};

const parseBlogPost = (source: string): BlogPost => {
  const { frontmatter, body } = parseFrontmatter(source);
  const rawMeta = Object.fromEntries(
    Object.entries(FRONTMATTER_KEYS).map(([sourceKey, targetKey]) => [
      targetKey,
      frontmatter[sourceKey],
    ])
  );

  for (const key of requiredMetaKeys) {
    if (!rawMeta[key]) {
      throw new Error(`Blogpost frontmatter is missing "${key}".`);
    }
  }

  const { nodes, footnotes } = parseNodes(body);

  return {
    meta: {
      title: rawMeta.title,
      subtitle: rawMeta.subtitle,
      byline: rawMeta.byline.split('|').map((entry) => entry.trim()),
    },
    nodes,
    footnotes,
  };
};

const ESCAPED_UNDERSCORE_PLACEHOLDER = '\uE000';

const restoreEscapedUnderscores = (text: string) =>
  text.replace(new RegExp(ESCAPED_UNDERSCORE_PLACEHOLDER, 'g'), '_');

const renderInlineMarkdown = (
  text: string,
  footnoteNumbersById?: ReadonlyMap<string, number>
): ReactNode[] => {
  const escapedText = text.replace(/\\_/g, ESCAPED_UNDERSCORE_PLACEHOLDER);
  const parts = escapedText.split(
    /(`[^`]+`|\[\^[^\]]+\]|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_)/
  );

  return parts
    .filter((part) => part.length > 0)
    .map((part, index) => {
      if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
        return <code key={index}>{part.slice(1, -1)}</code>;
      }

      const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        return (
          <a
            key={index}
            href={linkMatch[2]}
            target="_blank"
            rel="noreferrer"
          >
            {restoreEscapedUnderscores(linkMatch[1])}
          </a>
        );
      }

      const footnoteReferenceMatch = part.match(/^\[\^([^\]]+)\]$/);
      if (footnoteReferenceMatch) {
        const footnoteNumber = footnoteNumbersById?.get(
          footnoteReferenceMatch[1].trim()
        );
        if (footnoteNumber !== undefined) {
          return (
            <sup key={index} className="footnote-reference">
              <a href={`#footnote-${footnoteNumber}`}>{footnoteNumber}</a>
            </sup>
          );
        }
      }

      if (
        ((part.startsWith('**') && part.endsWith('**')) ||
          (part.startsWith('__') && part.endsWith('__'))) &&
        part.length >= 4
      ) {
        return (
          <strong key={index}>
            {restoreEscapedUnderscores(part.slice(2, -2))}
          </strong>
        );
      }

      if (
        ((part.startsWith('*') && part.endsWith('*')) ||
          (part.startsWith('_') && part.endsWith('_'))) &&
        part.length >= 3
      ) {
        return (
          <em key={index}>{restoreEscapedUnderscores(part.slice(1, -1))}</em>
        );
      }

      return <Fragment key={index}>{restoreEscapedUnderscores(part)}</Fragment>;
    });
};

export const buildPromptSource = (nodes: BlogNode[]) =>
  nodes
    .map((node) => {
      if (node.type === 'heading') {
        return `${'#'.repeat(node.level)} ${node.text}`;
      }

      if (
        node.type === 'paragraph' ||
        node.type === 'rewrite' ||
        node.type === 'callout' ||
        node.type === 'links'
      ) {
        return node.type === 'links' ? node.items.join(' ') : node.text;
      }

      return '';
    })
    .filter(Boolean)
    .join('\n\n')
    .trim();

export const blogPost = parseBlogPost(rawBlogpost);
export { renderInlineMarkdown };
