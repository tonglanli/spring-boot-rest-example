package com.khoubyari.example.api.rest;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;

import com.khoubyari.example.domain.Hotel;
import com.khoubyari.example.exception.DataFormatException;
import com.khoubyari.example.service.HotelService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import jcuda.*;
import jcuda.runtime.*;
import jcuda.driver.*;

/*
 * Demonstrates how to set up RESTful API endpoints using Spring MVC
 */

@RestController
@RequestMapping(value = "/")
@Api()
public class HotelController extends AbstractRestHandler {

    @Autowired
    private HotelService hotelService;

    @RequestMapping(value = "",
            method = RequestMethod.POST,
            consumes = {"application/json", "application/xml"},
            produces = {"application/json", "application/xml"})
    @ResponseStatus(HttpStatus.CREATED)
    @ApiOperation(value = "Create a hotel resource.", notes = "Returns the URL of the new resource in the Location header.")
    public void createHotel(@RequestBody Hotel hotel,
                            HttpServletRequest request, HttpServletResponse response) {
        Hotel createdHotel = this.hotelService.createHotel(hotel);
        response.setHeader("Location", request.getRequestURL().append("/").append(createdHotel.getId()).toString());
    }

    @RequestMapping(value = "",
            method = RequestMethod.GET,
            produces = {"application/json", "application/xml"})
    @ResponseStatus(HttpStatus.OK)
    @ApiOperation(value = "Get a paginated list of all hotels.", notes = "The list is paginated. You can provide a page number (default 0) and a page size (default 100)")
    public
    @ResponseBody
    Page<Hotel> getAllHotel(@ApiParam(value = "The page number (zero-based)", required = true)
                            @RequestParam(value = "page", required = true, defaultValue = DEFAULT_PAGE_NUM) Integer page,
                            @ApiParam(value = "Tha page size", required = true)
                            @RequestParam(value = "size", required = true, defaultValue = DEFAULT_PAGE_SIZE) Integer size,
                            HttpServletRequest request, HttpServletResponse response) {
        return this.hotelService.getAllHotels(page, size);
    }

    @RequestMapping(value = "/{id}",
            method = RequestMethod.GET,
            produces = {"application/json", "application/xml"})
    @ResponseStatus(HttpStatus.OK)
    @ApiOperation(value = "Get a single hotel.", notes = "You have to provide a valid hotel ID.")
    public
    @ResponseBody
    Hotel getHotel(@ApiParam(value = "The ID of the hotel.", required = true)
                   @PathVariable("id") Long id,
                   HttpServletRequest request, HttpServletResponse response) throws Exception {
        Hotel hotel = this.hotelService.getHotel(id);
        checkResourceFound(hotel);
        //todo: http://goo.gl/6iNAkz
        return hotel;
    }

    @RequestMapping(value = "/{id}",
            method = RequestMethod.PUT,
            consumes = {"application/json", "application/xml"},
            produces = {"application/json", "application/xml"})
    @ResponseStatus(HttpStatus.NO_CONTENT)
    @ApiOperation(value = "Update a hotel resource.", notes = "You have to provide a valid hotel ID in the URL and in the payload. The ID attribute can not be updated.")
    public void updateHotel(@ApiParam(value = "The ID of the existing hotel resource.", required = true)
                            @PathVariable("id") Long id, @RequestBody Hotel hotel,
                            HttpServletRequest request, HttpServletResponse response) {
        checkResourceFound(this.hotelService.getHotel(id));
        if (id != hotel.getId()) throw new DataFormatException("ID doesn't match!");
        this.hotelService.updateHotel(hotel);
    }

    //todo: @ApiImplicitParams, @ApiResponses
    @RequestMapping(value = "/{id}",
            method = RequestMethod.DELETE,
            produces = {"application/json", "application/xml"})
    @ResponseStatus(HttpStatus.NO_CONTENT)
    @ApiOperation(value = "Delete a hotel resource.", notes = "You have to provide a valid hotel ID in the URL. Once deleted the resource can not be recovered.")
    public void deleteHotel(@ApiParam(value = "The ID of the existing hotel resource.", required = true)
                            @PathVariable("id") Long id, HttpServletRequest request,
                            HttpServletResponse response) {
        checkResourceFound(this.hotelService.getHotel(id));
        this.hotelService.deleteHotel(id);
    }

    @RequestMapping(value = "/healthcheck",
            method = RequestMethod.GET,
            produces = {"application/json", "application/xml"})
    @ResponseStatus(HttpStatus.OK)
    @ApiOperation(value = "Check the health of the service.", notes = "This API will return 'success' if the service is running properly.")
    public String healthCheck() {
        return "";
    }

    public static class VectorAddRequest {

        public VectorAddRequest() {
        }

        private float[] hostX;
        private float[] hostY;

        // getters and setters

        public float[] getHostX() {
            return hostX;
        }

        public void setHostX(float[] hostX) {
            this.hostX = hostX;
        }

        public float[] getHostY() {
            return hostY;
        }

        public void setHostY(float[] hostY) {
            this.hostY = hostY;
        }
    }

    @RequestMapping(value = "/vectoradd",
            method = RequestMethod.POST,
            produces = {"application/json"})
    @ResponseStatus(HttpStatus.OK)
    @ApiOperation(value = "Perform vector addition using JCuda.", notes = "This API will return the result of vector addition.")
    public float[] vectorAdd(@RequestBody VectorAddRequest request) {
        float[] hostX = request.getHostX();
        float[] hostY = request.getHostY();
        float[] hostZ = new float[hostX.length];
        long startTime, endTime;
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);

        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "vectorAdd.ptx");

        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "add");

        int n = hostX.length;

        CUdeviceptr deviceX = new CUdeviceptr();
        CUdeviceptr deviceY = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        JCudaDriver.cuMemAlloc(deviceY, n * Sizeof.FLOAT);

        JCudaDriver.cuMemcpyHtoD(deviceX, Pointer.to(hostX), n * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoD(deviceY, Pointer.to(hostY), n * Sizeof.FLOAT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(deviceX),
                Pointer.to(deviceY)
        );

        startTime = System.nanoTime();
        JCudaDriver.cuLaunchKernel(function,
                (n + 1023)/1024,  1, 1,      // Grid dimension
                1024, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();
        endTime = System.nanoTime();
        System.out.println("JCuda Execution time: " + (endTime - startTime) + " ns");

        JCudaDriver.cuMemcpyDtoH(Pointer.to(hostY), deviceY, n * Sizeof.FLOAT);

        startTime = System.nanoTime();
        for (int i = 0; i < n; i++) {
            hostZ[i] = hostX[i] + hostY[i];
        }
        endTime = System.nanoTime();
        System.out.println("Java Execution time: " + (endTime - startTime) + " ns");

        JCudaDriver.cuMemFree(deviceX);
        JCudaDriver.cuMemFree(deviceY);
        JCudaDriver.cuCtxDestroy(context);

        return hostY;
    }
}
