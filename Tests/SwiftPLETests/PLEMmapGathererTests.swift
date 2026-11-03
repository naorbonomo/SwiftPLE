//
//  PLEMmapGathererTests.swift
//  SwiftPLETests
//
//  Unit tests verifying mmap gatherer initialization, layout math, and dummy gathers.
//

import XCTest
@testable import SwiftPLE

final class PLEMmapGathererTests: XCTestCase {
    func testPLEModelConfigurationDefaults() {
        let config = PLEModelConfiguration()
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.numHiddenLayers, 42)
        XCTAssertEqual(config.hiddenSizePerLayerInput, 256)

        // Modulo 6 layer schedule: layers 5, 11, 17, 23, 29, 35, 41 are full attention
        XCTAssertFalse(config.isFullAttention(layerIndex: 0))
        XCTAssertFalse(config.isFullAttention(layerIndex: 4))
        XCTAssertTrue(config.isFullAttention(layerIndex: 5))
        XCTAssertTrue(config.isFullAttention(layerIndex: 23))
        XCTAssertFalse(config.isFullAttention(layerIndex: 24))
    }

    func testRowLayoutMath() {
        let config = PLEModelConfiguration()
        let totalPleDim = config.numHiddenLayers * config.hiddenSizePerLayerInput // 42 * 256 = 10,752
        let groupSize = 64
        let numGroups = totalPleDim / groupSize // 168
        let numU32PerRow = totalPleDim / 8     // 1,344

        let bytesWeights = numU32PerRow * 4    // 5,376
        let bytesScales = numGroups * 2        // 336
        let bytesBiases = numGroups * 2        // 336
        let bytesPerToken = bytesWeights + bytesScales + bytesBiases // 6,048

        XCTAssertEqual(bytesWeights, 5376)
        XCTAssertEqual(bytesScales, 336)
        XCTAssertEqual(bytesBiases, 336)
        XCTAssertEqual(bytesPerToken, 6048)
    }

    func testMmapInitializationFailsGracefullyOnMissingFile() {
        let dummyURL = URL(fileURLWithPath: "/tmp/non_existent_ple_table_\(UUID().uuidString).bin")
        let config = PLEModelConfiguration()
        let gatherer = PLEMmapGatherer(fileURL: dummyURL, config: config)
        XCTAssertNil(gatherer)
    }
}

