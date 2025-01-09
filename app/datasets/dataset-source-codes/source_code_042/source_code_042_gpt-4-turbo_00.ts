class WYSIWYGEditor {
    private content: string;

    constructor(initialContent: string) {
        this.content = initialContent;
    }

    // Applies the specified Markdown transformation to the selected text
    applyMarkdownTransformation(start: number, end: number, style: 'bold' | 'italic'): void {
        const selectedText = this.content.substring(start, end);
        const beforeText = this.content.substring(0, start);
        const afterText = this.content.substring(end);

        const markdownOpen = style === 'bold' ? '**' : '*';
        const markdownClose = style === 'bold' ? '**' : '*';

        // Deal with potential partial Markdown tags within the selection
        const cleanSelected = this.removePartialMarkdown(selectedText);

        // Apply Markdown tags
        const styledText = `${markdownOpen}${cleanSelected}${markdownClose}`;

        this.content = `${beforeText}${styledText}${afterText}`;
    }

    // Cleans partial markdown tags within the selection
    private removePartialMarkdown(text: string): string {
        return text
            .replace(/\*\*/g, '') // Remove existing bold markdown within selected text
            .replace(/\*/g, '');  // Remove existing italic markdown within selected text
    }

    // Return the current state of the editor content
    getContent(): string {
        return this.content;
    }
}

// Example usage:
const editor = new WYSIWYGEditor('This is **an** example');
editor.applyMarkdownTransformation(5, 14, 'bold');
console.log(editor.getContent()); // "This **is an ex**ample"
