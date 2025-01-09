// within UserController.php 

use Illuminate\Http\Request;
use App\Models\User;
use Illuminate\Database\Eloquent\ModelNotFoundException;
  
class UserController extends Controller
{
    public function index(Request $request)
    {
        try {
            $query = User::query();

            if ($request->get('username')) {
                $query->where('username', 'like', '%' . $request->get('username') . '%');
            }
            
            $users = $query->paginate(10); // get 10 users per page

            if(!isset($users)){
                throw new ModelNotFoundException('No users found');
            }

            return response()->json($users, 200);
        } catch (ModelNotFoundException $exception) {
            return response()->json(["message" => $exception->getMessage()], 404);
        } catch (\Exception $exception) {
            return response()->json(["message" => "Server error, please try again later."], 500);
        }
    }
}
