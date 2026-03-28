import assert from "node:assert/strict";
import test from "node:test";
import {
  comparableStartsWith,
  longestComparablePrefixLength,
  shouldResetNoisyPartial,
} from "../src/utils/partialStability.ts";

test("comparableStartsWith ignores punctuation changes inside the prefix", () => {
  assert.equal(
    comparableStartsWith(
      "这张图把问题指清楚了，不是酒包回流，而是后续结果。把前面又重说了一遍",
      "这张图把问题指清楚了，不是酒包回流而是后续结果把前面又重说",
    ),
    true,
  );
});

test("longestComparablePrefixLength ignores punctuation and spacing noise", () => {
  assert.equal(
    longestComparablePrefixLength("第一段，已提交", "第一段 已提交，新增内容"),
    "第一段已提交".length,
  );
});

test("shouldResetNoisyPartial allows replacing a short mixed-script false start", () => {
  assert.equal(
    shouldResetNoisyPartial("you然", "这张图把问题指清楚了"),
    true,
  );
});

test("shouldResetNoisyPartial allows replacing a short trailing fragment with a full sentence", () => {
  assert.equal(
    shouldResetNoisyPartial(
      "那本",
      "这张图把问题指清楚了，不是酒包回流，而是后续结果把前面又重说了一遍。但是中间夹杂了少量识别纠错。导入之前那本呃精确前缀裁剪失手。",
    ),
    true,
  );
});

test("shouldResetNoisyPartial does not reset a normal longer prefix", () => {
  assert.equal(
    shouldResetNoisyPartial("这张图把问题指清楚了", "这是另一句完全不同的话"),
    false,
  );
});
