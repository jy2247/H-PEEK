#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
peek_repo="${1:-${project_root}/third_party/peek}"
expected_sha="3aca72b63c1dd6433193c0857b965acecdbfbb8a"
patch_file="${project_root}/patches/h-peek-history.patch"

actual_sha="$(git -C "${peek_repo}" rev-parse HEAD)"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  echo "PEEK revision mismatch: expected ${expected_sha}, got ${actual_sha}" >&2
  exit 1
fi

if [[ -n "$(git -C "${peek_repo}" status --porcelain)" ]]; then
  echo "PEEK worktree is not clean; refusing to mix the H-PEEK patch with other changes." >&2
  exit 1
fi

git -C "${peek_repo}" apply --unidiff-zero --check "${patch_file}"
git -C "${peek_repo}" apply --unidiff-zero "${patch_file}"
echo "Applied H-PEEK history patch to ${peek_repo}"
