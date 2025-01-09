import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Firestore Cache Invalidation',
      home: MessageScreen(),
    );
  }
}

class MessageScreen extends StatefulWidget {
  @override
  _MessageScreenState createState() => _MessageScreenState();
}

class _MessageScreenState extends State<MessageScreen> {
  List<String> _messages = [];
  DateTime _lastFetchTime;

  @override
  void initState() {
    super.initState();
    _fetchMessages();
  }

  Future<void> _fetchMessages() async {
    if (_lastFetchTime == null || DateTime.now().difference(_lastFetchTime) > Duration(minutes: 30)) {
      // Fetch new data from Firestore if cache is expired or not yet initialized
      final snapshot = await FirebaseFirestore.instance.collection('messages').get();
      setState(() {
        _messages = snapshot.docs.map((doc) => doc['message']).toList();
        _lastFetchTime = DateTime.now();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Messages'),
      ),
      body: Center(
        child: _messages.isEmpty
            ? CircularProgressIndicator() // Show loading indicator if messages are being fetched
            : ListView.builder(
                itemCount: _messages.length,
                itemBuilder: (context, index) => ListTile(
                  title: Text(_messages[index]),
                ),
              ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _fetchMessages,
        tooltip: 'Refresh',
        child: Icon(Icons.refresh),
      ),
    );
  }
}