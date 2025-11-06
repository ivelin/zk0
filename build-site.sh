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

if [ "$MODE" = "serve" ]; then
  # Copy root files for processing in serve mode
  rm -rf docs/ README.md CONTRIBUTING.md LICENSE .env.example
  cp -r ../docs ./docs/
  cp ../README.md .
  cp ../CONTRIBUTING.md .
  cp ../LICENSE .
  cp ../.env.example .

  # Serve with watch mode for development (no clean, incremental builds)
  bundle exec jekyll serve --source . --verbose
else
  # Clean previous builds
  rm -rf ../_site/ ../.jekyll-cache/

  # Copy root docs and other .md files to source for processing
  rm -rf docs/ README.md CONTRIBUTING.md LICENSE .env.example
  cp -r ../docs ./docs/
  cp ../README.md .
  cp ../CONTRIBUTING.md .
  cp ../LICENSE .
  cp ../.env.example .

  # Build with explicit source/destination
  bundle exec jekyll build --source . --destination ../_site --verbose

  cd .. || true

  echo "Jekyll build complete. Generated files in _site/ with URLs using https://zk0.bot"
  echo "Verify: cat _site/sitemap.xml"
fi