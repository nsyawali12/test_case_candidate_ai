import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class Message {
  final String text;
  final DateTime timestamp;

  Message(this.text, this.timestamp);

  Message.fromSnapshot(DocumentSnapshot snapshot)
      : text = snapshot['text'],
        timestamp = snapshot['timestamp'].toDate();
}

class CacheInvalidationModel extends ChangeNotifier {
  static const cacheTimeout = const Duration(minutes: 30);
  DateTime _lastFetchTimestamp;
  List<Message> _messages = [];

  List<Message> get messages => _messages;

  Future<void> fetchMessages() async {
    final currentTime = DateTime.now();

    if (_lastFetchTimestamp != null) {
      final timeDifference = currentTime.difference(_lastFetchTimestamp);

      if (timeDifference.compareTo(cacheTimeout) < 0) {
        // We have a valid cached value - don't do anything
        return;
      }
    }

    _lastFetchTimestamp = currentTime;

    // Fetch the list of messages from your Cloud Firestore instance
    _messages = [];

    var collection = FirebaseFirestore.instance.collection('messages');
    var snapshot = await collection.get();
    for (var docSnapshot in snapshot.docs) {
      _messages.add(Message.fromSnapshot(docSnapshot));
    }

    // Notify any listeners that the messages have changed
    notifyListeners();
  }
}
