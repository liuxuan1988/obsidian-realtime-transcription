import type { TranscriptionResult } from "../types";

const COMPARISON_SKIP_RE = /[\s\p{P}\p{S}]/u;
const LEADING_SEPARATOR_RE = /^[\s，。！？、,.;:：；!?\-—]+/u;

export interface TrimCommittedPrefixResult {
  hasOverlap: boolean;
  isDuplicate: boolean;
  trimmedText: string;
  shouldResetCommitted: boolean;
}

interface ComparableText {
  normalized: string;
  originalCutPoints: number[];
}

interface OverlapMatch {
  committedLength: number;
  incomingLength: number;
}

function toComparableText(text: string): ComparableText {
  let normalized = "";
  const originalCutPoints: number[] = [];

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (COMPARISON_SKIP_RE.test(char)) continue;
    normalized += char.toLowerCase();
    originalCutPoints.push(index + 1);
  }

  return { normalized, originalCutPoints };
}

function findApproxPrefixOverlap(committed: string, incoming: string): OverlapMatch {
  if (!committed || !incoming) {
    return { committedLength: 0, incomingLength: 0 };
  }

  const maxEdits = Math.max(1, Math.min(8, Math.floor(Math.min(committed.length, incoming.length) * 0.08)));
  let committedIndex = 0;
  let incomingIndex = 0;
  let edits = 0;
  let strongCommittedIndex = 0;
  let strongIncomingIndex = 0;
  let exactMatchesSinceEdit = 0;
  let pendingEdit = false;

  while (committedIndex < committed.length && incomingIndex < incoming.length) {
    if (committed[committedIndex] === incoming[incomingIndex]) {
      committedIndex += 1;
      incomingIndex += 1;
      if (pendingEdit) {
        exactMatchesSinceEdit += 1;
        if (exactMatchesSinceEdit >= 2) {
          strongCommittedIndex = committedIndex;
          strongIncomingIndex = incomingIndex;
          pendingEdit = false;
        }
      } else {
        strongCommittedIndex = committedIndex;
        strongIncomingIndex = incomingIndex;
      }
      continue;
    }

    if (edits >= maxEdits) break;

    const canSkipCommitted =
      committedIndex + 1 < committed.length &&
      committed[committedIndex + 1] === incoming[incomingIndex];
    const canSkipIncoming =
      incomingIndex + 1 < incoming.length &&
      committed[committedIndex] === incoming[incomingIndex + 1];
    const canSubstitute =
      committedIndex + 1 < committed.length &&
      incomingIndex + 1 < incoming.length &&
      committed[committedIndex + 1] === incoming[incomingIndex + 1];

    edits += 1;
    pendingEdit = true;
    exactMatchesSinceEdit = 0;

    if (canSubstitute) {
      committedIndex += 1;
      incomingIndex += 1;
      continue;
    }

    if (canSkipCommitted && !canSkipIncoming) {
      committedIndex += 1;
      continue;
    }

    if (canSkipIncoming) {
      incomingIndex += 1;
      continue;
    }

    committedIndex += 1;
    incomingIndex += 1;
  }

  if (pendingEdit) {
    return {
      committedLength: strongCommittedIndex,
      incomingLength: strongIncomingIndex,
    };
  }

  return {
    committedLength: strongCommittedIndex,
    incomingLength: strongIncomingIndex,
  };
}

export function trimCommittedPrefix(
  committedTexts: string[],
  incomingText: string,
): TrimCommittedPrefixResult {
  const committed = toComparableText(committedTexts.join(""));
  const incoming = toComparableText(incomingText);

  if (!committed.normalized || !incoming.normalized) {
    return {
      hasOverlap: false,
      isDuplicate: false,
      trimmedText: incomingText.trim(),
      shouldResetCommitted: false,
    };
  }

  const overlap = findApproxPrefixOverlap(committed.normalized, incoming.normalized);
  const overlapLength = overlap.committedLength;

  const hasMeaningfulOverlap = overlapLength >= committed.normalized.length * 0.5;
  if (!hasMeaningfulOverlap) {
    return {
      hasOverlap: false,
      isDuplicate: false,
      trimmedText: incomingText.trim(),
      shouldResetCommitted: overlapLength < 3 && incoming.normalized.length >= 4,
    };
  }

  const remainingNormalized = incoming.normalized.slice(overlap.incomingLength);
  if (!remainingNormalized || remainingNormalized.length < 2) {
    return {
      hasOverlap: true,
      isDuplicate: true,
      trimmedText: "",
      shouldResetCommitted: false,
    };
  }

  const cutIndex = incoming.originalCutPoints[overlap.incomingLength - 1] ?? 0;
  const trimmedText = incomingText.slice(cutIndex).replace(LEADING_SEPARATOR_RE, "").trim();

  return {
    hasOverlap: true,
    isDuplicate: false,
    trimmedText,
    shouldResetCommitted: false,
  };
}

export function isStalePartialResult(
  result: Pick<TranscriptionResult, "type" | "flush_seq">,
  currentFlushSeq: number,
): boolean {
  return result.type === "partial"
    && typeof result.flush_seq === "number"
    && result.flush_seq < currentFlushSeq;
}
