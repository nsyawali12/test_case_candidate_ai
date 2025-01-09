import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MessageCacheManager extends ChangeNotifier {
  List<String> _cachedMessages = [];
  DateTime _lastFetchTime;

  Future<void fetchMessages() async {
    if (_lastFetchTime != null && DateTime.now().difference(_lastFetchTime) < Duration(minutes: 30)) {
      return; // No need to fetch new data, using cached data
    }

    // Fetch new data from Firestore
    List<String> newData = await _fetchDataFromFirestore();
    _cachedMessages = newData;
    _lastFetchTime = DateTime.now();

    notifyListeners();
  }

  Future<List<String>> _fetchDataFromFirestore() async {
    // Simulated function to fetch data from Cloud Firestore
    // Replace this with actual Firestore call in your Flutter app
    // For demonstration purposes, a simple list is returned here
    return ['Message 1', 'Message 2', 'Message 3'];
  }

  List<String> getCachedMessages() {
    return _cachedMessages;
  }
}

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (context) => MessageCacheManager(),
      child: MaterialApp(
        home: Scaffold(
          appBar: AppBar(
            title: Text('Message App'),
          ),
          body: MessageList(),
        ),
      ),
    ),
  );
}

class MessageList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final cacheManager = Provider.of<MessageCacheManager>(context);

    return Column(
      children: <Widget>[
        RaisedButton(
          onPressed: () => cacheManager.fetchMessages(),
          child: Text('Fetch Messages'),
        ),
        SizedBox(height: 20),
        ListView.builder(
          shrinkWrap: true,
          itemCount: cacheManager.getCachedMessages().length,
          itemBuilder: (context, index) {
            return ListTile(
              title: Text(cacheManager.getCachedMessages()[index]),
            );
          },
        ),
      ],
    );
  }
}
