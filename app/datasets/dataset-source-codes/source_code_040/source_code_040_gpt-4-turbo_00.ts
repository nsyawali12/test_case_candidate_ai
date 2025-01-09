interface Task {
    id: number;
    title: string;
    parent_id: number | null;
}

interface NestedTask extends Task {
    children?: NestedTask[];
}

function buildTaskHierarchy(tasks: Task[]): NestedTask[] {
    const taskMap = new Map<number, NestedTask>();
    const result: NestedTask[] = [];

    // First pass to create a map of tasks and initialize children arrays
    tasks.forEach(task => {
        const nestedTask: NestedTask = { ...task, children: [] };
        taskMap.set(task.id, nestedTask);
    });

    // Second pass to populate the tree
    tasks.forEach(task => {
        const currentTask = taskMap.get(task.id)!;
        if (task.parent_id === null) {
            result.push(currentTask); // It's a root node
        } else {
            const parentTask = taskMap.get(task.parent_id);
            parentTask?.children.push(currentTask); // Nest it under its parent
        }
    });

    return result;
}
