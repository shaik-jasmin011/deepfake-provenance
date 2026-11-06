// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MediaRegistry {
    struct Record {
        bytes32 fileHash;
        bool isFake;
        uint16 fakeScoreBps;
        address registrar;
        uint256 timestamp;
    }

    mapping(bytes32 => Record) private records;
    mapping(bytes32 => bool) private registered;

    event MediaRegistered(
        bytes32 indexed fileHash,
        bool isFake,
        uint16 fakeScoreBps,
        address indexed registrar,
        uint256 timestamp
    );

    function registerMedia(bytes32 fileHash, bool isFake, uint16 fakeScoreBps) external {
        require(!registered[fileHash], "Media already registered");
        require(fakeScoreBps <= 10000, "Fake score exceeds 10000 bps");

        registered[fileHash] = true;
        records[fileHash] = Record({
            fileHash: fileHash,
            isFake: isFake,
            fakeScoreBps: fakeScoreBps,
            registrar: msg.sender,
            timestamp: block.timestamp
        });

        emit MediaRegistered(fileHash, isFake, fakeScoreBps, msg.sender, block.timestamp);
    }

    function getRecord(bytes32 fileHash) external view returns (Record memory) {
        require(registered[fileHash], "Media not registered");
        return records[fileHash];
    }

    function isRegistered(bytes32 fileHash) external view returns (bool) {
        return registered[fileHash];
    }
}
