class MarkdownEditor {
  private editorState: string;
  private undoStack: string[];

  constructor(initialContent: string) {
    this.editorState = initialContent;
    this.undoStack = [initialContent];
  }

  applyMarkdownTransformation(selectedText: string, style: 'bold' | 'italic'): void {
    const transformedText = this.getTransformedText(selectedText, style);
    const updatedContent = this.editorState.replace(selectedText, transformedText);
    
    this.updateEditorState(updatedContent);
  }

  private getTransformedText(selectedText: string, style: 'bold' | 'italic'): string {
    if (this.isPartOfMarkdownTag(selectedText)) {
      return selectedText;
    }

    // Apply the specified Markdown style to the selected text
    switch (style) {
      case 'bold':
        return `**${selectedText}**`;
      case 'italic':
        return `*${selectedText}*`;
      default:
        return selectedText;
    }
  }

  private isPartOfMarkdownTag(text: string): boolean {
    return text.startsWith('**') && text.endsWith('**') ||
           text.startsWith('*') && text.endsWith('*');
  }

  private updateEditorState(newContent: string): void {
    this.undoStack.push(this.editorState);
    this.editorState = newContent;
  }

  undo(): void {
    if (this.undoStack.length > 1) {
      this.editorState = this.undoStack.pop()!;
    }
  }
}

// Example of using the MarkdownEditor class
const editor = new MarkdownEditor('This is **an** example');
editor.applyMarkdownTransformation('is **an** ex', 'bold');
console.log(editor.editorState); // Output: 'This **is an ex**ample'
