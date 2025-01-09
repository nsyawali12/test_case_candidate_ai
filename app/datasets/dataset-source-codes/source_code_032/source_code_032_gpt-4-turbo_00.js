// Laravel Controller Method
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Database\Eloquent\ModelNotFoundException;

class UserController extends Controller
{
    public function getUsers(Request $request)
    {
        try {
            // Get page and username from query parameters
            $username = $request->query('username', '');
            $pageSize = $request->query('size', 15); // default page size to 15

            // Query using Eloquent with conditions and pagination
            $users = User::where('username', 'LIKE', "%$username%")->paginate($pageSize);

            return response()->json($users);
        } catch (\Exception $e) {
            Log::error('Failed to fetch users:', ['error' => $e->getMessage()]);
            // Return a general HTTP 500 response if there is an error
            return response()->json(['error' => 'Failed to fetch users'], 500);
        }
    }
}
