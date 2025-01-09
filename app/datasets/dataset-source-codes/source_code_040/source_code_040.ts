interface Input {
  id: number;
  title: string;
  parent_id: number | null;  
}

interface Output extends Input {
  children?: Output[];  
}

function doJob(inputItems: Input[], parent_id?: number) {
  const outputItems: Output[] = [];

  for (let i = 0; i < inputItems.length; i++) {
    const children = doJob(inputItems.slice(i, inputItems.length), inputItems[i].parent_id)
    .filter(i => i.parent_id === parent_id);
    
    outputItems.push({...item, children});
  }

  return outputItems;
}