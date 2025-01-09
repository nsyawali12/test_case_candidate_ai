interface State {
    editorContent: string;
    past: string[];
    future: string[];
}

const initialState: State = {
    editorContent: '',
    past: [],
    future: [],
};

function undo(state: State): State {
    if (state.past.length === 0) return state;
    const past = [...state.past];
    const editorContent = past[past.length - 1];
    past.pop();
    return {...state, editorContent, past, future: [state.editorContent, ...state.future]};
}

function redo(state: State): State {
    if (state.future.length === 0) return state;
    const future = [...state.future];
    const editorContent = future[0];
    future.shift();
    return {...state, editorContent, past: [...state.past, state.editorContent], future};
}

function applyStyle(state: State, selectedText: string, style: string): State {
    let styleTag = '**'; // by default bold
    if (style === 'italic') styleTag = '_'; // if italic is chosen
    const editorContent = state.editorContent.replace(selectedText, `${styleTag}${selectedText}${styleTag}`);
    return {
        ...state,
        editorContent,
        past: [...state.past, state.editorContent],
        future: [],
    };
}

let editorState = {...initialState, editorContent: 'This is **an** example'};

editorState = applyStyle(editorState, 'is **an** ex', 'bold');

console.log(editorState.editorContent);
// handles the undo
editorState = undo(editorState);
// handles the redo
editorState = redo(editorState);
