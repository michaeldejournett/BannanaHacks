# How to Load and View the zyBook Documentation

## Option 1: View Locally in Your Browser/IDE

### Using VS Code (Recommended)
1. Open the file in VS Code: `BannanaHacks/Docs/ZYBOOK_DOCUMENTATION.md`
2. Install the "Markdown Preview Enhanced" extension
3. Right-click the file → "Markdown Preview Enhanced: Open Preview"
4. Code blocks with `runnable` attribute may show execution buttons

### Using a Local Markdown Server
```bash
# Install a markdown viewer (if not installed)
pip install mkdocs-material

# Or use a simple HTTP server
cd /home/michaeldejournett/BananaHacks2/BannanaHacks/Docs
python3 -m http.server 8000
# Then open http://localhost:8000/ZYBOOK_DOCUMENTATION.md in your browser
```

### Using GitHub/GitLab
1. Push the file to a GitHub/GitLab repository
2. View it directly in the browser - GitHub renders markdown nicely
3. Code blocks will be syntax-highlighted (but may not be executable)

## Option 2: Import into zyBooks Platform

If you're using the zyBooks educational platform:

### Step 1: Create a Custom Section
1. Log into your zyBooks account
2. Navigate to your zyBook's home page
3. Expand the chapter where you want to add content
4. Scroll to the bottom and click "Create section"
5. Enter a title (e.g., "BannanaHacks Complete Guide")

### Step 2: Add Content Blocks
1. Use the formatting toolbar to add content blocks
2. For **text content**: Copy sections from the markdown file
3. For **code blocks**: Use "Code block" for syntax highlighting
4. For **runnable code**: Use "Code editor" content block type

### Step 3: Convert Markdown to zyBooks Format
Since zyBooks doesn't directly import markdown, you'll need to:

1. **Copy text sections**: Copy headings and paragraphs directly
2. **Add code blocks manually**: 
   - Select "Code block" from content type menu
   - Paste code from markdown file
   - Select appropriate language (bash/python)
3. **Add interactive code editors**:
   - Select "Code editor" content block
   - Paste code that should be runnable
   - This creates executable code blocks

### Step 4: Structure Your Content
- Break the document into multiple sections (one per major topic)
- Use zyBooks' section structure:
  - Part 1: Project Setup
  - Part 2: AMD GPU Configuration
  - Part 3: Training Your Model
  - etc.

## Option 3: Convert to HTML/PDF

### Convert to HTML
```bash
# Using pandoc (if installed)
cd /home/michaeldejournett/BananaHacks2/BannanaHacks/Docs
pandoc ZYBOOK_DOCUMENTATION.md -o ZYBOOK_DOCUMENTATION.html --standalone --toc

# Or using markdown-pdf
npm install -g markdown-pdf
markdown-pdf ZYBOOK_DOCUMENTATION.md
```

### View HTML in Browser
```bash
# Open the HTML file
xdg-open ZYBOOK_DOCUMENTATION.html  # Linux
# or
open ZYBOOK_DOCUMENTATION.html      # macOS
```

## Option 4: Use Interactive Markdown Viewers

### Using Jupyter Notebook
```bash
# Convert markdown to Jupyter notebook
pip install notedown
notedown ZYBOOK_DOCUMENTATION.md > ZYBOOK_DOCUMENTATION.ipynb

# Open in Jupyter
jupyter notebook ZYBOOK_DOCUMENTATION.ipynb
```

### Using Obsidian or Notion
- **Obsidian**: Open the markdown file directly - it renders beautifully
- **Notion**: Import the markdown file - it converts automatically

## Quick Start: View Right Now

The simplest way to view it immediately:

```bash
# Option A: View in VS Code
code /home/michaeldejournett/BananaHacks2/BannanaHacks/Docs/ZYBOOK_DOCUMENTATION.md

# Option B: View in browser (if you have a markdown viewer extension)
# Most modern browsers can render markdown with extensions

# Option C: Use GitHub
# Push to GitHub and view there - best markdown rendering
```

## Notes for zyBooks Import

When importing into zyBooks:
- Code blocks with `runnable` attribute → Use "Code editor" block type
- Regular code blocks → Use "Code block" with syntax highlighting
- Text content → Copy directly into text blocks
- Tables → Use zyBooks table tool
- Headings → Use zyBooks heading styles

The markdown file is structured to be easily copy-pasted into zyBooks sections.

