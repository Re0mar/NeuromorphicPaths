plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.neuromorphicpaths.pixel"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.neuromorphicpaths.pixel"
        minSdk = 34
        targetSdk = 37
        versionCode = 1
        versionName = "0.1"
        // The Pixel 8 and the x86_64 emulator. Nothing else carries this.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    testOptions {
        // JVM unit tests run against android.jar stubs that throw on every call. The connection
        // logs through android.util.Log from its catch blocks, and a stub that throws there kills
        // the thread the test is watching. Defaults make Log a no-op off the device.
        unitTests.isReturnDefaultValues = true
    }
    sourceSets {
        getByName("main").kotlin.srcDir("src/main/kotlin")
        getByName("test").kotlin.srcDir("src/test/kotlin")
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.arcore)

    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.org.json)
}

// The encoder test writes the frame the laptop's suite decodes. Point it at that suite's fixtures
// directory with -PwireFixturePath=... so one run regenerates the committed fixture.
tasks.withType<Test>().configureEach {
    systemProperty("wireFixturePath", (project.findProperty("wireFixturePath") as String?) ?: "build/pixel_app_frame.bin")
}
