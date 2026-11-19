#!/usr/bin/env bash
set -euo pipefail

# cleanup_git_artifacts.sh
#
# Removes generated artefacts from Git tracking WITHOUT deleting local files.
#
# Usage:
#   ./cleanup_git_artifacts.sh          # dry run
#   ./cleanup_git_artifacts.sh --apply  # actually untrack files

APPLY=false

if [[ "${1:-}" == "--apply" ]]; then
    APPLY=true
fi

echo "=== Git generated-artifact cleanup ==="
echo

# Patterns I am fairly confident should not be version controlled.
PATTERNS=(
    '*/source_snapshot/*'
    '*/frozen_runtime/*'
    '*.npy'
    '*.pkl'
    '*.err'
    '*.out'
    '*/tests.xml'
)

# Find tracked files matching the patterns.
tmpfile="$(mktemp)"
trap 'rm -f "$tmpfile"' EXIT

for pattern in "${PATTERNS[@]}"; do
    git ls-files "$pattern"
done | sort -u > "$tmpfile"

count=$(wc -l < "$tmpfile" | tr -d ' ')

echo "Found $count currently tracked files matching cleanup rules."
echo

if [[ "$count" -eq 0 ]]; then
    echo "Nothing to clean."
    exit 0
fi

echo "Examples:"
head -n 40 "$tmpfile"

if [[ "$count" -gt 40 ]]; then
    echo
    echo "... plus $((count - 40)) more"
fi

echo

if [[ "$APPLY" == false ]]; then
    echo "DRY RUN ONLY."
    echo
    echo "Review the list with:"
    echo
    echo "  cat $tmpfile"
    echo
    echo "If it looks sensible, run:"
    echo
    echo "  $0 --apply"
    echo
    exit 0
fi

echo "Removing these files from Git tracking..."
echo "(The files remain on disk.)"

while IFS= read -r file; do
    git rm --cached --ignore-unmatch -- "$file"
done < "$tmpfile"

echo
echo "Adding ignore rules to .gitignore..."

cat >> .gitignore <<'EOF'

# ------------------------------------------------------------
# Generated study / runtime artefacts
# ------------------------------------------------------------

**/source_snapshot/
**/frozen_runtime/

*.npy
*.pkl

*.err
*.out
**/tests.xml
EOF

# Avoid duplicate .gitignore lines if script gets rerun.
awk '!seen[$0]++ || $0 == ""' .gitignore > .gitignore.tmp
mv .gitignore.tmp .gitignore

echo
echo "Done."
echo
echo "IMPORTANT: Nothing has been committed yet."
echo
echo "Now inspect:"
echo
echo "  git status"
echo "  git diff --cached --stat"
echo "  git diff --cached --name-status"
echo
echo "If happy:"
echo
echo '  git add .gitignore'
echo '  git commit -m "Remove generated study artifacts from repository"'
echo '  git push origin main'