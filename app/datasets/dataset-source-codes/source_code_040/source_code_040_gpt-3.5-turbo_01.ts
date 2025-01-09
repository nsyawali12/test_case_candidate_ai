interface Task {
    id: number;
    title: string;
    parent_id: number | null;
    children?: Task[];
}

function convertToTree(tasks: Task[]): Task[] {
    const taskMap = tasks.reduce((map, task) => {
        map[task.id] = { ...task, children: [] };
        return map;
    }, {});

    const rootTasks: Task[] = [];

    tasks.forEach(task => {
        if (task.parent_id === null) {
            rootTasks.push(taskMap[task.id]);
        } else {
            const parentTask = taskMap[task.parent_id];
            if (parentTask) {
                parentTask.children?.push(taskMap[task.id]);
            }
        }
    });

    return rootTasks;
}

const input: Task[] = [
    { id: 1, title: 'Task 1', parent_id: null },
    { id: 2, title: 'Task 2', parent_id: 1 },
    { id: 3, title: 'Task 3', parent_id: 1 }
];

const output = convertToTree(input);
console.log(output);
