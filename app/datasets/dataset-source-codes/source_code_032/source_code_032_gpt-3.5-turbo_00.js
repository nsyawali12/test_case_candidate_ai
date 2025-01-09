// Laravel - UserController.php

use App\Models\User;
use Illuminate\Http\Request;

class UserController extends Controller
{
    public function getUsers(Request $request)
    {
        $perPage = 10; // Number of users per page

        $users = User::query();

        if ($request->has('username')) {
            $users->where('username', 'like', '%'.$request->username.'%');
        }

        $users = $users->paginate($perPage);

        return response()->json($users);
    }
}
