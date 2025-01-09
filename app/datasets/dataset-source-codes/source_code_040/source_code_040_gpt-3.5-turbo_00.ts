interface Task {
    id: number;
    title: string;
    parent_id: number | null;
    children?: Task[];
}

function buildTaskTree(tasks: Task[]): Task[] {
    const tasksMap = tasks.reduce((acc, task) => {
        acc[task.id] = { ...task, children: [] };
        return acc;
    }, {} as { [key: number]: Task });

    const rootTasks: Task[] = [];

    tasks.forEach(task => {
        const { id, parent_id } = task;
        if (parent_id === null) {
            rootTasks.push(tasksMap[id]);
        } else {
            if (tasksMap[parent_id]) {
                tasksMap[parent_id].children.push(tasksMap[id]);
            }
        }
    });

    return rootTasks;
}

// Test the function with the provided example
const inputTasks: Task[] = [
    { id: 1, title: 'Task 1', parent_id: null },
    { id: 2, title: 'Task 2', parent_id: 1 },
    { id: 3, title: 'Task 3', parent_id: 1 }
];
const outputTasks: Task[] = buildTaskTree(inputTasks);
console.log(outputTasks);
