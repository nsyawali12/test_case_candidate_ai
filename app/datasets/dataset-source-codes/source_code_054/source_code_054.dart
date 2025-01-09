import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Realtime Database Demo',
      home: MessageScreen(),
    );
  }
}

class MessageScreen extends StatelessWidget {
  final DatabaseReference _databaseRef = FirebaseDatabase.instance.reference().child('messages');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Messages')),
      body: Column(
        children: [
          Expanded(
            child: StreamBuilder(
              stream: _databaseRef.orderByChild('timestamp').onValue,
              builder: (context, snapshot) {
                if (snapshot.hasError || snapshot.connectionState == ConnectionState.waiting)
                  return Center(child: CircularProgressIndicator());
                List messages = snapshot.data.snapshot.value?.values?.toList() ?? [];
                messages.sort((a, b) => b['timestamp'].compareTo(a['timestamp']));
                return ListView.builder(
                  itemCount: messages.length,
                  itemBuilder: (context, index) => ListTile(
                    title: Text(messages[index]['text']),
                    subtitle: Text(messages[index]['timestamp']),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    decoration: InputDecoration(hintText: 'Enter message'),
                    onSubmitted: (text) => _addMessage(text),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send),
                  onPressed: () => _addMessage(text),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _addMessage(String text) {
    if (text.trim().isEmpty) return;
    _databaseRef.push().set({'text': text, 'timestamp': DateTime.now().toUtc().toString()});
  }
}