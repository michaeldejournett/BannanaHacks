# Markdown Preview Enhanced - Execution Setup Guide

## Enable Code Execution in MPE

To see run buttons in Markdown Preview Enhanced, you need to:

### Step 1: Enable Code Chunk Execution
1. Open your markdown file in VS Code
2. Open Markdown Preview Enhanced (right-click → "Markdown Preview Enhanced: Open Preview")
3. Right-click in the preview pane
4. Go to **Settings** → **Code Chunk Settings**
5. Enable **"Enable Code Chunk Execution"**
6. Enable **"Enable Script Execution"** (for bash scripts)

### Step 2: Configure Execution Settings
In MPE settings, you may need to:
- Set **"Shell"** to `/bin/bash` (or your shell path)
- Set **"Python"** to your Python interpreter path
- Enable **"Auto Execute"** if you want code to run automatically

### Step 3: Alternative Format
If run buttons still don't appear, try this format in your markdown:

```markdown
```{.bash .run}
# Your code here
```
```

Or use code chunks with execution:

```markdown
```{.bash .run .cmd="bash"}
# Your code here
```
```

### Step 4: Check MPE Version
Make sure you have the latest version of Markdown Preview Enhanced:
- Open VS Code Extensions
- Search for "Markdown Preview Enhanced"
- Update if available

### Troubleshooting
If run buttons still don't appear:
1. **Reload VS Code**: `Ctrl+Shift+P` → "Developer: Reload Window"
2. **Check MPE Settings**: Right-click preview → Settings → Code Chunk
3. **Try different format**: Some MPE versions use `{.bash}` while others need `{.bash .run}`
4. **Check console**: Open VS Code Developer Tools (Help → Toggle Developer Tools) to see errors

### Manual Execution
If automatic execution doesn't work, you can:
- Right-click on code blocks in the preview
- Select "Run Code Chunk" from the context menu
- Or copy-paste code directly into terminal

