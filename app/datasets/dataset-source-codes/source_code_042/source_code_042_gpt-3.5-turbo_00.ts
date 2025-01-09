interface EditorState {
  content: string;
  selectionStart: number;
  selectionEnd: number;
}

class WysiwygEditor {
  private state: EditorState;
  private history: EditorState[];

  constructor(initialContent: string) {
    this.state = { content: initialContent, selectionStart: 0, selectionEnd: 0 };
    this.history = [this.state];
  }

  applyMarkdownStyle(selectedText: string, style: 'bold' | 'italic'): void {
    const { content, selectionStart, selectionEnd } = this.state;
    const startTag = style === 'bold' ? '**' : '_';
    const endTag = startTag;

    const beforeSelection = content.slice(0, selectionStart);
    const selectedTextWithTags = `${startTag}${selectedText}${endTag}`;
    const afterSelection = content.slice(selectionEnd);

    const updatedContent = beforeSelection + selectedTextWithTags + afterSelection;
    
    this.updateState({ content: updatedContent, selectionStart: 0, selectionEnd: 0 });
  }

  updateState(newState: EditorState): void {
    this.state = newState;
    this.history.push(this.state);
  }

  undo(): void {
    if (this.history.length > 1) {
      this.history.pop();
      this.state = this.history[this.history.length - 1];
    }
  }
}

// Example usage
const editor = new WysiwygEditor("This is **an** example");
editor.applyMarkdownStyle("is **an** ex", "bold");
console.log(editor.state.content); // Output: "This **is an ex**ample"
