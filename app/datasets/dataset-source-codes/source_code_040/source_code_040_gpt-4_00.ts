// Define the Task interface. TypeScript interfaces are needed to define the type of object.
interface Task {
  id: number;
  title: string;
  parent_id: number | null;
  children?: Task[];
}

// Function buildTree converts a flat array of tasks into a nested structure
function buildTree(tasks: Task[]): Task[] {
  // Object to hold tasks by their id for quick lookup
  let tasksMap: { [key: number]: Task & { children: Task[] } } = {};

  // First pass: create a copy of tasks and map them by id
  for (let task of tasks) {
    tasksMap[task.id] = { ...task, children: [] };
  }

  // Array to hold the root tasks that have no parent
  let rootTasks: Task[] = [];

  // Second pass: add children tasks to their respective parents
  for (let task of tasks) {
    // If task has a parent
    if (task.parent_id !== null) {
      // Add this task to its parent's children array
      tasksMap[task.parent_id].children.push(tasksMap[task.id]);
    } else {
      // If no parent_id, add task to root tasks
      rootTasks.push(tasksMap[task.id]);
    }
  }

  // Return the nested task tree
  return rootTasks;
}
