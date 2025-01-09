// Import necessary packages
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_database/firebase_database.dart';

// Initialize Firebase
void initializeFirebase() async {
  await Firebase.initializeApp();
}

// Class representing the Flutter app widget
class MessageWidget extends StatelessWidget {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final DatabaseReference _database = FirebaseDatabase().reference().child('messages');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Real-time Messages'),
      ),
      body: StreamBuilder(
        stream: _database.orderByChild('timestamp').onValue,
        builder: (context, snapshot) {
          if (snapshot.hasData && snapshot.data != null) {
            // Convert retrieved data to a List
            List<dynamic> messages = [];
            snapshot.data.snapshot.value.forEach((key, value) {
              messages.add(value);
            });

            messages.sort((a, b) => a['timestamp'].compareTo(b['timestamp']));

            return ListView.builder(
              itemCount: messages.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(messages[index]['text']),
                  subtitle: Text(messages[index]['timestamp']),
                );
              },
            );
          } else if (snapshot.hasError) {
            return Text('Error retrieving messages');
          } else {
            return Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // Add a new message to the database
          _database.push().set({
            'text': 'New Message',
            'timestamp': DateTime.now().toUtc().toString(),
          });
        },
        child: Icon(Icons.add),
      ),
    );
  }
}

void main() {
  initializeFirebase();
  runApp(MaterialApp(home: MessageWidget()));
}
