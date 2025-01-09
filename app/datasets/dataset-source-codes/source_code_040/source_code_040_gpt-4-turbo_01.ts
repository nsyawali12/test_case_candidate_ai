interface Task {
    id: number;
    title: string;
    parent_id: number | null;
}

interface NestedTask extends Task {
    children: NestedTask[];
}

function buildNestedTasks(tasks: Task[]): NestedTask[] {
    const taskMap: Map<number, NestedTask> = new Map();

    // Create a map of tasks and initialize children array
    tasks.forEach(task => {
        taskMap.set(task.id, {...task, children: []});
    });

    // Result array for root tasks
    const roots: NestedTask[] = [];

    // Build the tree structure
    taskMap.forEach((nestedTask, id) => {
        if (nestedTask.parent_id === null) {
            roots.push(nestedTask); // Root task
        } else {
            const parentTask = taskMap.get(nestedTask.parent_id);
            if (parentTask) {
                parentTask.children.push(nestedTask); // Add to parent task's children
            }
        }
    });

    return roots;
}
