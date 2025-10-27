# Label Studio Setup Guide

## Step-by-Step Import Instructions

### 1. Create New Project
- Click "Create Project"
- Project Name: **SMS Phishing Annotation**

### 2. Import Pre-Annotated Data
- Click "Settings" (gear icon)
- Click "Import" tab on the left
- Click "Upload Files"
- Select: `data/annotations/preannotated.json`
- **IMPORTANT**: Check the box "Treat CSV/TSV as List of tasks"
- Click "Import"

### 3. Configure Labeling Interface
- In Settings, click "Labeling Interface"
- Click "Code" button (top right corner)
- Paste this configuration:

```xml
<View>
  <Header value="Annotate SMS Entities"/>
  <Text name="text" value="$text"/>
  
  <Labels name="label" toName="text">
    <Label value="BRAND" background="#FF6B6B" hotkey="b"/>
    <Label value="PHONE" background="#4ECDC4" hotkey="p"/>
    <Label value="URL" background="#45B7D1" hotkey="u"/>
    <Label value="ORDER_ID" background="#96CEB4" hotkey="o"/>
    <Label value="AMOUNT" background="#FFEAA7" hotkey="m"/>
    <Label value="DATE" background="#DFE6E9" hotkey="d"/>
    <Label value="DEADLINE" background="#FF7675" hotkey="l"/>
    <Label value="ACTION_REQUIRED" background="#FD79A8" hotkey="a"/>
  </Labels>
</View>
```

- Click "Save"

### 4. Enable Predictions Display
- Go back to project main page
- Click on first task
- In the annotation interface:
  - Look for "Predictions" dropdown (top right)
  - Select "gpt4-mini-batch-v1"
  - The AI annotations should now appear highlighted!

### 5. Annotation Workflow
For each message:
1. Review the pre-highlighted entities
2. Click highlighted regions to:
   - **Accept** (if correct)
   - **Edit** boundaries (drag edges)
   - **Delete** (click X)
3. Add missing entities:
   - Select text with mouse
   - Press hotkey (b/p/u/o/m/d/l/a)
4. Click "Submit" when complete

## Troubleshooting

**Problem**: Predictions not showing
- Solution: Click "Predictions" dropdown and select "gpt4-mini-batch-v1"

**Problem**: Can't see highlighted text
- Solution: Make sure labeling config has correct `name="text"` and `value="$text"`

**Problem**: Wrong entity types
- Solution: Click entity → Change label in dropdown

## Keyboard Shortcuts
- `b` = BRAND
- `p` = PHONE  
- `u` = URL
- `o` = ORDER_ID
- `m` = AMOUNT
- `d` = DATE
- `l` = DEADLINE
- `a` = ACTION_REQUIRED
- `Ctrl+Enter` = Submit
- `Ctrl+Space` = Show predictions

## Export Annotations When Done
Settings → Export → JSON or CoNLL format
