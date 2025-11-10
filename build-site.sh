#!/bin/bash
# Build Jekyll site from website/ subdir to avoid config loading issues
# Usage: ./build-site.sh [--dev|--serve]
#   --dev|--serve: Run in development mode with jekyll serve for local testing (watches files, serves on http://localhost:4000)

set -e  # Exit on error

# Parse arguments
MODE="build"
while [[ $# -gt 0 ]]; do
  case $1 in
    --dev|--serve)
      MODE="serve"
      shift
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: ./build-site.sh [--dev|--serve]"
      exit 1
      ;;
  esac
done

if [ "$MODE" = "serve" ]; then
  echo "Starting Jekyll development server..."
else
  echo "Building Jekyll site..."
fi

cd website/ || { echo "Failed to cd to website/"; exit 1; }

# Copy root docs/ into website/docs/ for Jekyll processing (Markdown to HTML conversion)
echo "Copying root docs/ to website/docs/..."
rm -rf docs/  # Clean previous copy
mkdir -p docs/
cp -r ../docs/* docs/ 2>/dev/null || echo "No docs/ to copy or copy failed (non-fatal for initial setup)"

# Copy root LICENSE and CONTRIBUTING.md for site links (static and rendered)
echo "Copying root LICENSE and CONTRIBUTING.md..."
cp ../LICENSE ./ 2>/dev/null || echo "LICENSE copy failed (non-fatal)"
cp ../CONTRIBUTING.md ./ 2>/dev/null || echo "CONTRIBUTING.md copy failed (non-fatal)"
ls -la LICENSE CONTRIBUTING.md || echo "Copied files verification failed"

if [ "$MODE" = "serve" ]; then
  # Serve with watch mode for development (no clean, incremental builds)
  # Note: For live reload of docs/, manually re-run or use a watcher; copy happens on each start
  bundle exec jekyll serve --source . --verbose
else
  # Clean previous builds
  rm -rf ../_site/ ../.jekyll-cache/
  # Re-copy docs/ after clean for fresh build
  rm -rf docs/
  mkdir -p docs/
  cp -r ../docs/* docs/ 2>/dev/null || echo "No docs/ to copy or copy failed (non-fatal)"

  # Re-copy root LICENSE and CONTRIBUTING.md after clean
  cp ../LICENSE ./ 2>/dev/null || echo "LICENSE copy failed (non-fatal)"
  cp ../CONTRIBUTING.md ./ 2>/dev/null || echo "CONTRIBUTING.md copy failed (non-fatal)"
  cp ../get-zk0bot.sh ./ 2>/dev/null || echo "get-zk0bot.sh copy failed (non-fatal)"
  ls -la LICENSE CONTRIBUTING.md get-zk0bot.sh || echo "Copied files verification failed"

  # Build with explicit source/destination
  bundle exec jekyll build --source . --destination ../_site --verbose

  cd .. || true

  echo "Jekyll build complete. Generated files in _site/ with URLs using https://zk0.bot"
  echo "Verify: cat _site/sitemap.xml"
fi