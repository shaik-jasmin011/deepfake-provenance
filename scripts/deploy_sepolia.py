"""Deploy MediaRegistry to Sepolia without exposing local credentials."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.blockchain import SEPOLIA_CHAIN_ID, ProvenanceClient

SEPOLIA_DEPLOYMENT_PATH = PROJECT_ROOT / "contracts" / "deployment_sepolia.json"
ETHERSCAN_ADDRESS_URL = "https://sepolia.etherscan.io/address/"
ETHERSCAN_TRANSACTION_URL = "https://sepolia.etherscan.io/tx/"


def _update_env_contract_address(address: str) -> None:
    env_path = PROJECT_ROOT / ".env"
    with env_path.open("r", encoding="utf-8", newline="") as env_file:
        lines = env_file.readlines()

    matching_lines = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n").startswith("CONTRACT_ADDRESS=")
    ]
    if len(matching_lines) != 1:
        raise ValueError(".env must contain exactly one CONTRACT_ADDRESS= line.")

    index = matching_lines[0]
    line_ending = lines[index][len(lines[index].rstrip("\r\n")) :]
    lines[index] = f"CONTRACT_ADDRESS={address}{line_ending}"
    with env_path.open("w", encoding="utf-8", newline="") as env_file:
        env_file.writelines(lines)


def _deploy() -> None:
    if not os.environ.get("RPC_URL") or not os.environ.get("PRIVATE_KEY"):
        raise ValueError("RPC_URL and PRIVATE_KEY must be configured in .env.")

    client = ProvenanceClient()
    chain_id = client.web3.eth.chain_id
    if chain_id != SEPOLIA_CHAIN_ID:
        raise RuntimeError(f"Connected chain ID {chain_id}; expected Sepolia.")
    if client.account is None:
        raise RuntimeError("Unable to derive the deployer account.")

    balance = client.web3.from_wei(
        client.web3.eth.get_balance(client.account.address),
        "ether",
    )
    print(f"Connected chain ID: {chain_id}")
    print(f"Deployer address: {client.account.address}")
    print(f"Deployer balance: {balance} ETH")

    address = client.deploy(persist=False)
    transaction_hash = client.last_deployment_tx_hash
    if transaction_hash is None:
        raise RuntimeError("Deployment completed without a transaction hash.")

    SEPOLIA_DEPLOYMENT_PATH.write_text(
        json.dumps({"address": address, "abi": client.abi}, indent=2) + "\n",
        encoding="utf-8",
    )
    _update_env_contract_address(address)

    print(f"Contract address: {address}")
    print(f"Contract explorer: {ETHERSCAN_ADDRESS_URL}{address}")
    print(f"Deployment transaction: {transaction_hash}")
    print(f"Deployment transaction explorer: {ETHERSCAN_TRANSACTION_URL}{transaction_hash}")


def main() -> int:
    try:
        _deploy()
    except Exception as error:
        print(f"Sepolia deployment failed ({type(error).__name__}).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
