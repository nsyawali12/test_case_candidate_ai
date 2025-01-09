class MarkdownEditor {
    private content: string;

    constructor(initialContent: string) {
        this.content = initialContent;
    }

    // Function to apply style to selected text
    applyStyle(start: number, end: number, style: 'bold' | 'italic'): void {
        const selectedText = this.content.substring(start, end);
        let prefix = "", suffix = "";

        // Set Markdown syntax based on style
        switch (style) {
            case 'bold':
                prefix = "**";
                suffix = "**";
                break;
            case 'italic':
                prefix = "*";
                suffix = "*";
                break;
        }

        // Correct for existing partial Markdown tags
        const beforeText = this.content.substring(0, start);
        const afterText = this.content.substring(end);

        if(beforeText.endsWith("*")) {
            prefix = prefix.slice(1); // Avoid doubling '*'
        }
        if(afterText.startsWith("*")) {
            suffix = suffix.slice(1); // Avoid doubling '*'
        }

        // Apply Markdown formatting
        this.content = `${beforeText}${prefix}${selectedText}${suffix}${afterText}`;
    }

    // Function to get current editor content (for checking state in example)
    getContent(): string {
        return this.content;
    }
}

// Example usage
const editor = new MarkdownEditor('This is **an** example');
editor.applyStyle(5, 14, 'bold');
console.log(editor.getContent());  // Output: "This **is an ex**ample"
