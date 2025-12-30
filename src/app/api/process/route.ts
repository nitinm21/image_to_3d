import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import crypto from "crypto";

// Modal endpoint URL - will be set after deployment
const MODAL_ENDPOINT_URL = process.env.MODAL_ENDPOINT_URL;

// Directory to store temporary PLY files
const PLY_DIR = path.join(process.cwd(), "public", "ply");

export async function POST(request: NextRequest) {
  try {
    // Get the image from the form data
    const formData = await request.formData();
    const imageFile = formData.get("image") as File | null;

    if (!imageFile) {
      return NextResponse.json(
        { success: false, error: "No image provided" },
        { status: 400 }
      );
    }

    // Validate file type
    const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
    if (!validTypes.includes(imageFile.type)) {
      return NextResponse.json(
        { success: false, error: "Invalid file type. Please use PNG, JPG, or WEBP." },
        { status: 400 }
      );
    }

    // Convert file to base64
    const arrayBuffer = await imageFile.arrayBuffer();
    const base64Image = Buffer.from(arrayBuffer).toString("base64");

    // Check if Modal endpoint is configured
    if (!MODAL_ENDPOINT_URL) {
      return NextResponse.json(
        {
          success: false,
          error: "Modal endpoint not configured. Please set MODAL_ENDPOINT_URL environment variable."
        },
        { status: 500 }
      );
    }

    // Call Modal endpoint
    const modalResponse = await fetch(MODAL_ENDPOINT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: base64Image,
      }),
    });

    if (!modalResponse.ok) {
      const errorText = await modalResponse.text();
      console.error("Modal API error:", errorText);
      return NextResponse.json(
        { success: false, error: "Failed to process image with SHARP model" },
        { status: 500 }
      );
    }

    const modalData = await modalResponse.json();

    if (!modalData.success) {
      return NextResponse.json(
        { success: false, error: modalData.error || "Processing failed" },
        { status: 500 }
      );
    }

    // Decode base64 PLY data
    const plyBuffer = Buffer.from(modalData.ply_base64, "base64");

    // Generate unique filename
    const fileId = crypto.randomBytes(8).toString("hex");
    const fileName = `scene_${fileId}.ply`;

    // Ensure PLY directory exists
    if (!existsSync(PLY_DIR)) {
      await mkdir(PLY_DIR, { recursive: true });
    }

    // Write PLY file to public directory
    const filePath = path.join(PLY_DIR, fileName);
    await writeFile(filePath, plyBuffer);

    // Return URL to the PLY file
    const plyUrl = `/ply/${fileName}`;

    return NextResponse.json({
      success: true,
      plyUrl: plyUrl,
      message: "3D scene generated successfully",
    });
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "An error occurred"
      },
      { status: 500 }
    );
  }
}
