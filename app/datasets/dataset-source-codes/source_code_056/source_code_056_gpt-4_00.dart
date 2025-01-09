import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class Message {
  final String text;
  final Timestamp timeStamp;
  Message(this.text, this.timeStamp);
}

class MessageCache with ChangeNotifier {
  List<Message> _cache = [];
  DateTime _lastFetchedTime;

  List<Message> get cache {
    return [..._cache];
  }

  Future<void> fetchMessages() async {
    // Check if we need to update the cache
    if (_lastFetchedTime == null || DateTime.now().difference(_lastFetchedTime).inMinutes > 30) {
      final data = await FirebaseFirestore.instance.collection('messages').get();

      _cache = data.docs.map((doc) => Message(doc['text'], doc['timeStamp'])).toList();

      // Keep track of when we last fetched the data
      _lastFetchedTime = DateTime.now();

      // Notify listeners, i.e. UI, to update
      notifyListeners();
    }
  }
}
