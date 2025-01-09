type EditorState = {
    content: string
};

type MarkdownStyle = "bold" | "italic";

/**
 * Encapsulate a given text to apply the specified markdown style.
 * @param text The text to transform.
 * @param style The markdown style to apply.
 * @returns The transformed text.
 */
function applyMarkdownStyle(text: string, style: MarkdownStyle): string {
    switch (style) {
        case "bold":
            return `**${text}**`;
        case "italic":
            return `*${text}*`;
    }
}

/**
 * A function to apply a markdown style to selected text in the editor's state.
 * Uses undoable state operations for change tracking.
 * @param editorState The editor's state.
 * @param history The editor state history.
 * @param selectedText The selected text to transform.
 * @param style The markdown style to apply.
 */
function transformSelectedText(
  editorState: EditorState,
  history: EditorState[],
  selectedText: string,
  style: MarkdownStyle
) {
    // Copy the editor's content to be able to restore it later if necessary.
    history.push({ content: editorState.content });

    // Remove any existing markdown tags from the selected text.
    const cleanSelectedText = selectedText.replace(/(\*|\*\*)/g, "");

    // Apply the markdown transformation to the clean selected text.
    const transformedText = applyMarkdownStyle(cleanSelectedText, style);

    // Replace the selected text by the transformed text in the editor's content.
    editorState.content = editorState.content.replace(selectedText, transformedText);
}

// Test the function with the example.
const initialContent = 'This is **an** example';

let editorState: EditorState = { content: initialContent };
let editorStateHistory: EditorState[] = [];

transformSelectedText(editorState, editorStateHistory, 'is **an** ex', 'bold');

console.log(editorState.content);  // Outputs: 'This **is an ex**ample'
