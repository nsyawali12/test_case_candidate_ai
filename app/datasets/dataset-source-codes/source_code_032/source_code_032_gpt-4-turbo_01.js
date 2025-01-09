// Laravel API Controller

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\User;
use Illuminate\Database\Eloquent\ModelNotFoundException;

class UserController extends Controller
{
    /**
     * Display a listing of the resource with pagination and optional username filtering.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function index(Request $request)
    {
        try {
            $username = $request->query('username');
            $query = User::query();
            
            if ($username) {
                $query->where('username', 'like', '%' . $username . '%');
            }

            $users = $query->paginate(10);
            return response()->json($users);
        } catch (\Exception $exception) {
            return response()->json(['error' => $exception->getMessage()], 500);
        }
    }
}
