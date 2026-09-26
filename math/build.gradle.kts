import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

// Bytecode level matches the Android modules, otherwise app cannot consume these classes.
java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

dependencies {
    api(project(":core"))
    testImplementation(libs.junit)
}

// Tools under the test source set re-run the field over logged replays. They are not tests, so
// the ordinary test task leaves them out and each has a task of its own that passes the paths in.
tasks.test {
    filter {
        excludeTestsMatching("*Tool")
    }
}

tasks.register<Test>("priorSweep") {
    description = "Re-runs the field over a replay log with several turn priors and scores each against the walker's real turns."
    group = "tools"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    filter {
        includeTestsMatching("*.PriorSweepTool")
    }
    outputs.upToDateWhen { false }
    for (name in listOf("replay.log", "orientation.csv", "out.dir", "video.offset.seconds")) {
        project.findProperty(name)?.let { systemProperty(name, it) }
    }
    testLogging {
        showStandardStreams = true
    }
}

tasks.register<Test>("noPath") {
    description = "Re-runs the field over a replay log and reports the surprise of the least surprising heading on each frame."
    group = "tools"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    filter {
        includeTestsMatching("*.NoPathTool")
    }
    outputs.upToDateWhen { false }
    project.findProperty("replay.log")?.let { systemProperty("replay.log", it) }
    testLogging {
        showStandardStreams = true
    }
}
