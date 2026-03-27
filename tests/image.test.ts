import { describe, expect, it } from "vitest";

import { normalizeProfileImageUrl } from "../src/shared/image-core";

describe("normalizeProfileImageUrl", () => {
  it("upgrades Twitter avatar sizes to 400x400", () => {
    expect(
      normalizeProfileImageUrl("https://pbs.twimg.com/profile_images/123/example_normal.jpg?foo=bar"),
    ).toBe("https://pbs.twimg.com/profile_images/123/example_400x400.jpg");
  });
});
