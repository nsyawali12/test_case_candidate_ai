interface ContentState {
  blocks: Block[];
}

interface Block {
  key: string;
  children?: Child[];
  type: string;
  data?: {
    markdown?: boolean;
    text?: string;
    bold?: boolean;
    italic?: boolean;
  };
}

interface Child {
  text: string;
  entityRanges: EntityRange[];
}

interface EntityRange {
  offset: number;
  length: number;
  key: string;
}

export default class App extends React.Component {
  private editorRef: React.RefObject<Editor>;
  private store: DraftLocalStorageState;

  constructor(props: Props) {
    super(props);
    this.editorRef = React.createRef();
    this.store = DraftLocalStorageState('draftjs-markdown-transform');

    this.state = {
      editorContent: '',
      selectedText: '',
      style: '' as 'bold' | 'italic',
    };

    this.handleChange = this.handleChange.bind(this);
    this.applyMarkdownTransform = this.applyMarkdownTransform.bind(this);
  }

  componentDidUpdate(_prevProps: Props, _prevState: Readonly<{}>, snapshot: Snapshot) {
    if (snapshot?.hasChanged('selectedText')) {
      this.applyMarkdownTransform();
    }
  }

  applyMarkdownTransform() {
    const { selectedText, style } = this.state;
    const { editorState } = this.editorRef.current!;
    const contentBlockMap = EditorState.contentBlocks(editorState);
    const transformedContent = transformSelectedText(contentBlockMap, selectedText, style);

    this.store.replace(convertFromRaw(transformedContent));
    this.setState({ editorContent: JSON.stringify(toJSON(this.store.createEmpty()), undefined, 2) });
  }

  // ...
}

function transformSelectedText(
  contentBlockMap: Map<string, Block>,
  selectedText: string,
  style: 'bold' | 'italic'
): ContentState {
  //....
}