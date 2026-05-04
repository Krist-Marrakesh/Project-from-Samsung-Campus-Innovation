plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.example.myapplication"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "com.example.myapplication"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Backend base URL is read at run-time from BuildConfig so the same
        // APK can target different backends without re-coding.
        //
        //   * AVD emulator     — 10.0.2.2 is the AVD alias for the host
        //                        loopback, works out of the box.
        //   * Physical device  — set FRACTALOV_BACKEND_URL env var (e.g.
        //                        http://192.168.1.42:8080) before
        //                        `./gradlew installDebug`. Or use
        //                        `adb reverse tcp:8080 tcp:8080` and pass
        //                        http://localhost:8080.
        val backendUrl = System.getenv("FRACTALOV_BACKEND_URL") ?: "http://10.0.2.2:8080"
        buildConfigField("String", "BACKEND_BASE_URL", "\"$backendUrl\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
}

dependencies {
    // Compose BOM keeps every UI library on a coherent version set.
    implementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(platform(libs.androidx.compose.bom))

    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.navigation.compose)
    debugImplementation(libs.androidx.compose.ui.tooling)

    // Ktor 3 client + content-negotiation + kotlinx.serialization.
    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.okhttp)
    implementation(libs.ktor.client.content.negotiation)
    implementation(libs.ktor.client.logging)
    implementation(libs.ktor.serialization.kotlinx.json)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)

    // Coil 3 — Compose-native image loading; we feed it ByteArray for the
    // base64-decoded PNG so it doesn't have to fetch anything itself.
    implementation(libs.coil.compose)

    // Legacy AppCompat / Material kept around because the launcher theme still
    // references AppCompat XML attributes; removing them is a separate cleanup.
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}
