"""Local Ethereum provenance client for image hashes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from solcx import compile_source, get_installed_solc_versions, install_solc
from web3 import EthereumTesterProvider, Web3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "contracts" / "MediaRegistry.sol"
DEPLOYMENT_PATH = PROJECT_ROOT / "contracts" / "deployment.json"
SOLC_VERSION = "0.8.20"


def _ensure_solc() -> None:
    if not any(str(version) == SOLC_VERSION for version in get_installed_solc_versions()):
        install_solc(SOLC_VERSION)


def _compile_contract() -> tuple[list[dict[str, Any]], str]:
    _ensure_solc()
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    compiled = compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
    )
    _, contract_data = next(iter(compiled.items()))
    return contract_data["abi"], contract_data["bin"]


def sha256_to_bytes32(sha256_hex: str) -> bytes:
    """Convert exactly 64 hexadecimal SHA-256 characters to a bytes32 value."""
    normalized = sha256_hex[2:] if sha256_hex.startswith(("0x", "0X")) else sha256_hex
    if len(normalized) != 64:
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal characters")
    try:
        value = bytes.fromhex(normalized)
    except ValueError as error:
        raise ValueError("SHA-256 must contain only hexadecimal characters") from error
    if len(value) != 32:
        raise ValueError("SHA-256 must decode to exactly 32 bytes")
    return value


class ProvenanceClient:
    """Deploy or connect to a MediaRegistry and submit/read image provenance."""

    def __init__(
        self,
        web3: Web3 | None = None,
        contract_address: str | None = None,
    ) -> None:
        rpc_url = os.environ.get("RPC_URL")
        if web3 is None:
            web3 = Web3(Web3.HTTPProvider(rpc_url)) if rpc_url else Web3(EthereumTesterProvider())
        self.web3 = web3
        self.private_key = os.environ.get("PRIVATE_KEY")
        self.account = None
        if self.private_key:
            self.account = self.web3.eth.account.from_key(self.private_key)
            self.web3.eth.default_account = self.account.address
        elif self.web3.eth.accounts:
            self.web3.eth.default_account = self.web3.eth.accounts[0]

        deployment = self._read_deployment()
        self.abi: list[dict[str, Any]] = deployment.get("abi", [])
        configured_address = (
            contract_address
            or os.environ.get("CONTRACT_ADDRESS")
            or deployment.get("contract_address")
        )
        self.contract_address = (
            self.web3.to_checksum_address(configured_address) if configured_address else None
        )
        self.contract = (
            self.web3.eth.contract(address=self.contract_address, abi=self.abi)
            if self.contract_address and self.abi
            else None
        )
        self._bytecode: str | None = None

    @staticmethod
    def _read_deployment() -> dict[str, Any]:
        if not DEPLOYMENT_PATH.is_file():
            return {}
        return json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))

    def deploy(self) -> str:
        """Compile/deploy the registry and persist its ABI/address for reuse."""
        self.abi, bytecode = _compile_contract()
        factory = self.web3.eth.contract(abi=self.abi, bytecode=bytecode)
        sender = self.account.address if self.account else self.web3.eth.default_account
        transaction_hash = factory.constructor().transact({"from": sender})
        receipt = self.web3.eth.wait_for_transaction_receipt(transaction_hash)
        self.contract_address = self.web3.to_checksum_address(receipt.contractAddress)
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.abi)
        DEPLOYMENT_PATH.write_text(
            json.dumps(
                {
                    "contract_address": self.contract_address,
                    "abi": self.abi,
                    "solc_version": SOLC_VERSION,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.contract_address

    def _require_contract(self) -> Any:
        if self.contract is None:
            raise RuntimeError(
                "MediaRegistry is not deployed. Deploy locally or set CONTRACT_ADDRESS "
                "to a deployed registry address."
            )
        return self.contract

    def register(self, sha256_hex: str, is_fake: bool, p_fake: float) -> str:
        file_hash = sha256_to_bytes32(sha256_hex)
        if not isinstance(is_fake, bool):
            raise ValueError("is_fake must be a boolean")
        if not math.isfinite(p_fake) or not 0 <= p_fake <= 1:
            raise ValueError("p_fake must be a finite probability between 0 and 1")
        contract = self._require_contract()
        sender = self.account.address if self.account else self.web3.eth.default_account
        transaction_hash = contract.functions.registerMedia(
            file_hash,
            is_fake,
            int(round(p_fake * 10000)),
        ).transact({"from": sender})
        receipt = self.web3.eth.wait_for_transaction_receipt(transaction_hash)
        if receipt.status != 1:
            raise RuntimeError("MediaRegistry transaction failed")
        return self.web3.to_hex(transaction_hash)

    def verify(self, sha256_hex: str) -> dict[str, Any]:
        file_hash = sha256_to_bytes32(sha256_hex)
        contract = self._require_contract()
        is_registered = contract.functions.isRegistered(file_hash).call()
        if not is_registered:
            return {
                "registered": False,
                "is_fake": None,
                "p_fake": None,
                "registrar": None,
                "timestamp": None,
            }
        record = contract.functions.getRecord(file_hash).call()
        return {
            "registered": True,
            "is_fake": bool(record[1]),
            "p_fake": int(record[2]) / 10000,
            "registrar": self.web3.to_checksum_address(record[3]),
            "timestamp": int(record[4]),
        }


def get_provenance_client() -> ProvenanceClient:
    """Return one client per process, deploying a fresh ephemeral local registry."""
    client = getattr(get_provenance_client, "_client", None)
    if client is None:
        client = ProvenanceClient()
        if not os.environ.get("RPC_URL") and not os.environ.get("CONTRACT_ADDRESS"):
            client.deploy()
        get_provenance_client._client = client
    return client
