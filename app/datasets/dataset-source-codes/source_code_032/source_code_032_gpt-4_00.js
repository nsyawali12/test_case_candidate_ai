<?php
namespace App\Http\Controllers;
use Illuminate\Http\Request;
use App\Models\User;
use Illuminate\Pagination\Paginator;

class UserController extends Controller
{
    public function getUsers(Request $request)
    {
        try {
            $username = $request->query('username');
            $page = $request->query('page', 1);
            Paginator::currentPageResolver(function () use ($page) {
                return $page;
            });

            $usersQuery = User::query();
            if ($username) {
                $usersQuery->where('username', 'like', '%' . $username . '%');
            }

            $users = $usersQuery->paginate(5);
            return response()->json($users);
        } catch (\Exception $e) {
            return response()->json(['error' => $e->getMessage()], 500);
        }
    }
}
