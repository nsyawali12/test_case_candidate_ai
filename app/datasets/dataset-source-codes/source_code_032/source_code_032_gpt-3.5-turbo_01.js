// UserController.php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\User;

class UserController extends Controller
{
    public function getUsers(Request $request)
    {
        $query = User::query();

        // Filtering by username if provided in the request
        if ($request->has('username')) {
            $query->where('username', 'like', '%'.$request->input('username').'%');
        }

        // Paginating the results
        $users = $query->paginate(10); // 10 users per page

        return response()->json($users);
    }
}
