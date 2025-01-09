api file
Route::get('/users',UserController@index)

UserController.php


Public function index(Request $request){
try{
            $query = User::query();
           if($request->has('username')){
          $query->where(usersname,like,% $request->input('username')%);
      }
          
       $users = $query->paginate(10)
       return response()->json($users);
      }
      catch(/Exception $e){
      return response()->json(['error'=>'error'],500);
      }
}


userList.file

<template>
<div>
<input v-model="filter" @input="fetchUsers">
<ul>
<li v-for="u in users">{{user.username}}</li>
</ul>
<pagination :data="users" @pagination-change-page="fetchUsers"></pagination>
</div>
</template>
<script>
export default{
data(){
users:[],
filter:''
}
mounted(){
this.fetchUsers();
},
method:{
fetchUsers(page = 1){
axios.get('/api/users?page=${page}&username=${this.filer}')
.then(res=>{
this.users = res.data.data
})
.catch(e=>{
console.log('Error')
})
}
}
}  
</script>