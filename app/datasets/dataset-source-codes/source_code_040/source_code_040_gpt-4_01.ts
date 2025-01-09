// Define the task interface
interface Task {
  id: number;
  title: string;
  parent_id: number | null;
  children?: Task[];
}

// Define the task tree builder function
function buildTaskTree(tasks: Task[]): Task[] {
  const taskMap: Record<number, Task> = {};

  // first pass: create a copy of each task and store it in a map
  tasks.forEach(task => {
    const { id, title, parent_id } = task;
    taskMap[id] = { id, title, parent_id, children: [] };
  });

  const rootTasks: Task[] = [];

  // second pass: populate the children array for each task
  tasks.forEach(task => {
    const clonedTask = taskMap[task.id];
    if (task.parent_id === null) {
      // this is a root task
      rootTasks.push(clonedTask);
    } else {
      // this task is a child of another task
      const parentTask = taskMap[task.parent_id];
      if (parentTask) {
        parentTask.children.push(clonedTask);
      }
    }
  });

  return rootTasks;
}

// usage
const input: Task[] = [
  { id: 1, title: 'Task 1', parent_id: null },
  { id: 2, title: 'Task 2', parent_id: 1 },
  { id: 3, title: 'Task 3', parent_id: 1 },
];
const output = buildTaskTree(input);
console.log(JSON.stringify(output, null, 2));
